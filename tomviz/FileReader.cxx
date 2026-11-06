/* This source file is part of the Tomviz project, https://tomviz.org/.
   It is released under the 3-Clause BSD License, see "LICENSE". */

#include "FileReader.h"

#include "ActiveObjects.h"
#include "DataExchangeFormat.h"
#include "EmdFormat.h"
#include "FileFormatManager.h"
#include "FxiFormat.h"
#include "GenericHDF5Format.h"
#include "LoadDataReaction.h"
#include "PythonReader.h"
#include "PythonUtilities.h"
#include "Utilities.h"
#include "vtkOMETiffReader.h"

#include <vtkImageData.h>
#include <vtkJPEGReader.h>
#include <vtkMRCReader.h>
#include <vtkNew.h>
#include <vtkPNGReader.h>
#include <vtkPythonInterpreter.h>
#include <vtkSMSessionProxyManager.h>
#include <vtkSMSourceProxy.h>
#include <vtkSmartPointer.h>
#include <vtkStringArray.h>
#include <vtkTIFFReader.h>
#include <vtkXMLImageDataReader.h>

#include <QFileInfo>

namespace tomviz {

namespace {

QString fileExtension(const QStringList& fileNames)
{
  if (fileNames.isEmpty()) {
    return {};
  }
  return QFileInfo(fileNames.first()).suffix().toLower();
}

/// Read .tif/.tiff/.png/.jpg/.jpeg/.vti/.mrc directly via VTK without
/// going through ParaView's file-load machinery — so no dialogs.
vtkSmartPointer<vtkImageData> readWithVTK(const QStringList& fileNames)
{
  if (fileNames.isEmpty()) {
    return nullptr;
  }
  QString ext = fileExtension(fileNames);
  std::string path = fileNames.first().toStdString();

  if (ext == "tif" || ext == "tiff") {
    auto reader = vtkSmartPointer<vtkTIFFReader>::New();
    if (fileNames.size() > 1) {
      auto sa = vtkSmartPointer<vtkStringArray>::New();
      sa->SetNumberOfValues(static_cast<vtkIdType>(fileNames.size()));
      for (int i = 0; i < fileNames.size(); ++i) {
        sa->SetValue(static_cast<vtkIdType>(i),
                     fileNames[i].toStdString());
      }
      reader->SetFileNames(sa);
    } else {
      reader->SetFileName(path.c_str());
    }
    reader->Update();
    return reader->GetOutput();
  }
  if (ext == "png") {
    auto reader = vtkSmartPointer<vtkPNGReader>::New();
    reader->SetFileName(path.c_str());
    reader->Update();
    return reader->GetOutput();
  }
  if (ext == "jpg" || ext == "jpeg") {
    auto reader = vtkSmartPointer<vtkJPEGReader>::New();
    reader->SetFileName(path.c_str());
    reader->Update();
    return reader->GetOutput();
  }
  if (ext == "vti") {
    auto reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(path.c_str());
    reader->Update();
    return reader->GetOutput();
  }
  return nullptr;
}

/// File extensions of MRC stacks and volumes (IMOD's names included).
bool isMrc(const QString& suffix)
{
  static const QStringList mrc = { "mrc", "mrcs", "st",
                                   "ali", "preali", "rec" };
  return mrc.contains(suffix);
}

/// Tilt angles for an MRC tilt series, from its extended header or the
/// IMOD/SerialEM files next to it (tomviz.io.mrc_tilts), or none.
QVector<double> mrcTiltAngles(const QString& fileName, int count)
{
  QVector<double> angles;
  if (!vtkPythonInterpreter::IsInitialized()) {
    return angles;
  }
  Python python;
  auto module = python.import("tomviz.io.mrc_tilts");
  if (!module.isValid()) {
    return angles;
  }
  auto function = module.findFunction("tilt_angles");
  if (!function.isValid()) {
    return angles;
  }
  Python::Tuple args(2);
  Python::Object file(fileName);
  args.set(0, file);
  args.set(1, Variant(count));
  auto res = function.call(args);
  if (!res.isValid() || !res.isList()) {
    Python::checkForPythonError();
    return angles;
  }
  auto list = res.toList();
  for (int i = 0; i < list.length(); ++i) {
    angles.append(list[i].toDouble());
  }
  return angles;
}

/// A NeXus NXtomo file (tomviz.io.formats.nxtomo): projections with
/// angles, plus flat and dark frames. False when the file is not NXtomo
/// (or Python is not running), so other HDF5 readers can try.
bool readNxtomo(const QString& fileName, ReadResult& result)
{
  if (!vtkPythonInterpreter::IsInitialized()) {
    return false;
  }
  Python python;
  auto module = python.import("tomviz.io.formats.nxtomo");
  if (!module.isValid()) {
    return false;
  }
  auto check = module.findFunction("is_nxtomo");
  if (!check.isValid()) {
    return false;
  }
  Python::Tuple args(1);
  Python::Object file(fileName);
  args.set(0, file);
  auto isNxtomo = check.call(args);
  if (!isNxtomo.isValid() || !isNxtomo.toBool()) {
    Python::checkForPythonError();
    return false;
  }
  auto cls = module.findFunction("NXtomoReader");
  auto instance = cls.isValid() ? cls.call() : Python::Object();
  if (!instance.isValid()) {
    Python::checkForPythonError();
    return false;
  }
  auto parts = PythonReader(instance).readAll(fileName);
  if (!parts.data) {
    return false;
  }
  result.imageData = parts.data;
  result.darkData = parts.dark;
  result.whiteData = parts.flat;
  return true;
}

/// An MRC file: the image (spacing converted from the Angstrom MRC uses to
/// the nanometres Tomviz uses) and, for a tilt series, its angles.
ReadResult readMrc(const QString& fileName)
{
  ReadResult result;
  auto reader = vtkSmartPointer<vtkMRCReader>::New();
  reader->SetFileName(fileName.toLocal8Bit().constData());
  reader->Update();
  vtkSmartPointer<vtkImageData> image = reader->GetOutput();
  if (!image || image->GetNumberOfPoints() == 0) {
    return result;
  }
  double spacing[3];
  image->GetSpacing(spacing);
  image->SetSpacing(spacing[0] * 0.1, spacing[1] * 0.1, spacing[2] * 0.1);
  result.imageData = image;
  int dims[3];
  image->GetDimensions(dims);
  // IMOD's .rec is a reconstructed volume; it often sits next to the
  // .rawtlt of its tilt series, so do not look for angles for it.
  const bool volume = QFileInfo(fileName).suffix().toLower() == "rec";
  if (dims[2] > 1 && !volume) {
    result.tiltAngles = mrcTiltAngles(fileName, dims[2]);
  }
  return result;
}

QVariantMap hdf5OptionsFromJson(const QJsonObject& options)
{
  QVariantMap hdf5Options;
  if (options.contains("subsampleSettings")) {
    hdf5Options["subsampleStrides"] =
      options.value("subsampleSettings").toObject()["strides"].toVariant();
    hdf5Options["subsampleVolumeBounds"] =
      options.value("subsampleSettings")
        .toObject()["volumeBounds"]
        .toVariant();
    hdf5Options["askForSubsample"] = false;
  }
  return hdf5Options;
}

} // namespace

namespace {
ReadResult readAllParts(const QStringList& fileNames,
                        const QJsonObject& options);
} // namespace

ReadResult readImageData(const QStringList& fileNames,
                         const QJsonObject& options)
{
  ReadResult result = readAllParts(fileNames, options);
  // options["part"] picks a file's flat or dark frames instead of its
  // data, so those can be datasets of their own that re-read themselves.
  const QString part = options.value("part").toString();
  if (part.isEmpty() || part == QLatin1String("data")) {
    return result;
  }
  ReadResult partResult;
  partResult.imageData =
    part == QLatin1String("flat") ? result.whiteData : result.darkData;
  partResult.readerProperties = result.readerProperties;
  return partResult;
}

namespace {

ReadResult readAllParts(const QStringList& fileNames,
                         const QJsonObject& options)
{
  ReadResult result;
  if (fileNames.isEmpty()) {
    return result;
  }

  const QString fileName = fileNames.first();
  QFileInfo info(fileName);
  const QString suffix = info.suffix().toLower();
  const QString completeSuffix = info.completeSuffix().toLower();

  // .tvh5 — requires an internal node path.
  if (suffix == "tvh5") {
    QString path = options.value("tvh5NodePath").toString();
    if (path.isEmpty()) {
      return result;
    }
    QVariantMap emdOptions = { { "askForSubsample", false } };
    vtkNew<vtkImageData> image;
    if (EmdFormat::readNode(fileName.toStdString(), path.toStdString(),
                            image, emdOptions)) {
      result.imageData = image;
      result.tvh5NodePath = path;
    }
    return result;
  }

  // .emd — simple EMD reader.
  if (suffix == "emd") {
    QVariantMap emdOptions = hdf5OptionsFromJson(options);
    vtkNew<vtkImageData> imageData;
    if (EmdFormat::read(fileName.toLatin1().data(), imageData, emdOptions)) {
      result.imageData = imageData;
    }
    return result;
  }

  // .hspy / .h5 / .nxs — DataExchange / FXI / generic HDF5. NeXus
  // files are HDF5 underneath, so they go through the same readers.
  if (suffix == "hspy" || suffix == "h5" || suffix == "nxs" ||
      suffix == "nx") {
    QVariantMap hdf5Options = hdf5OptionsFromJson(options);
    const auto fnStd = fileName.toStdString();
    if (readNxtomo(fileName, result)) {
      return result;
    }
    if (GenericHDF5Format::isDataExchange(fnStd)) {
      DataExchangeFormat format;
      auto r = format.readAll(fileName.toLatin1().data(), hdf5Options);
      if (r.imageData) {
        result.imageData = r.imageData;
        result.tiltAngles = r.tiltAngles;
        result.darkData = r.darkData;
        result.whiteData = r.whiteData;
      }
      return result;
    }
    if (GenericHDF5Format::isFxi(fnStd)) {
      FxiFormat format;
      auto r = format.readAll(fileName.toLatin1().data(), hdf5Options);
      if (r.imageData) {
        result.imageData = r.imageData;
        result.tiltAngles = r.tiltAngles;
        result.darkData = r.darkData;
        result.whiteData = r.whiteData;
      }
      return result;
    }
    vtkNew<vtkImageData> imageData;
    GenericHDF5Format hdf5Format;
    if (hdf5Format.read(fileName.toLatin1().data(), imageData, hdf5Options)) {
      result.imageData = imageData;
    }
    return result;
  }

  // *.ome.tif[f] — multi-channel OME-TIFF.
  if (completeSuffix.endsWith("ome.tif") ||
      completeSuffix.endsWith("ome.tiff")) {
    vtkNew<vtkOMETiffReader> reader;
    reader->SetFileName(fileName.toLocal8Bit().constData());
    reader->Update();
    result.imageData = reader->GetOutput();
    QJsonObject rprops;
    rprops["name"] = "OMETIFFReader";
    result.readerProperties = rprops;
    return result;
  }

  // MRC stacks and volumes, with tilt angles when they are a tilt series.
  if (isMrc(suffix)) {
    return readMrc(fileName);
  }

  // Simple VTK-readable formats — direct, no dialog.
  if (suffix == "tif" || suffix == "tiff" || suffix == "png" ||
      suffix == "jpg" || suffix == "jpeg" || suffix == "vti") {
    result.imageData = readWithVTK(fileNames);
    return result;
  }

  // Python-registered reader by extension.
  if (FileFormatManager::instance().pythonReaderFactory(suffix) != nullptr) {
    auto factory = FileFormatManager::instance().pythonReaderFactory(suffix);
    auto reader = factory->createReader();
    auto parts = reader.readAll(fileNames[0]);
    if (parts.data) {
      result.imageData = parts.data;
      result.darkData = parts.dark;
      result.whiteData = parts.flat;
    }
    return result;
  }

  // ParaView reader — requires an explicit descriptor from the caller
  // (no dialog here). The descriptor carries the reader's XML name and
  // property values, as emitted by LoadDataReaction::readerProperties().
  if (options.contains("reader")) {
    auto props = options.value("reader").toObject();
    auto name = props.value("name").toString();
    auto* pxm = ActiveObjects::instance().proxyManager();
    if (!pxm || name.isEmpty()) {
      return result;
    }
    vtkSmartPointer<vtkSMProxy> reader;
    reader.TakeReference(pxm->NewProxy("sources", name.toLatin1().data()));
    if (!reader) {
      return result;
    }
    tomviz::setProperties(props, reader);
    LoadDataReaction::setFileNameProperties(props, reader);
    reader->UpdateVTKObjects();
    if (auto* src = vtkSMSourceProxy::SafeDownCast(reader)) {
      src->UpdatePipelineInformation();
      src->UpdatePipeline();
      auto* algo = vtkAlgorithm::SafeDownCast(src->GetClientSideObject());
      if (algo) {
        auto* image =
          vtkImageData::SafeDownCast(algo->GetOutputDataObject(0));
        if (image) {
          result.imageData = image;
          result.readerProperties = props;
        }
      }
    }
    return result;
  }

  // No headless path — caller must fall back to interactive load.
  return result;
}

} // namespace

} // namespace tomviz
