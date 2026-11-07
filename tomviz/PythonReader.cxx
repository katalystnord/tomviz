/* This source file is part of the Tomviz project, https://tomviz.org/.
   It is released under the 3-Clause BSD License, see "LICENSE". */

#include "PythonReader.h"

#include <vtkImageData.h>

#include <QDebug>

namespace tomviz {

PythonReader::PythonReader(Python::Object instance) : m_instance(instance)
{
}

PythonReader::PythonReader()
{
}

vtkSmartPointer<vtkImageData> PythonReader::read(QString fileName)
{
  return readAll(fileName).data;
}

namespace {
vtkSmartPointer<vtkImageData> toImage(Python::Object obj)
{
  if (!obj.isValid()) {
    return nullptr;
  }
  vtkObjectBase* vtkobject =
    Python::VTK::GetPointerFromObject(obj, "vtkImageData");
  return vtkImageData::SafeDownCast(vtkobject);
}
} // namespace

PythonReadResult PythonReader::readAll(QString fileName)
{
  PythonReadResult result;
  if (!m_instance.isValid()) {
    qWarning() << "The Python reader for this file type hasn't loaded yet. "
                  "Please try again in a few seconds";
    return result;
  }

  Python python;
  auto module = python.import("tomviz.io._internal");
  if (!module.isValid()) {
    qCritical() << "Failed to import tomviz.io._internal module.";
    return result;
  }

  auto readerFunction = module.findFunction("execute_reader");
  if (!readerFunction.isValid()) {
    qCritical() << "Failed to import tomviz.io._internal.execute_reader";
    return result;
  }

  Python::Tuple args(2);
  Python::Object argInstance = m_instance;
  args.set(0, argInstance);
  Python::Object fileArg(fileName);
  args.set(1, fileArg);

  auto res = readerFunction.call(args);
  if (!res.isValid() || !res.isList()) {
    qCritical("Error calling the reader");
    return result;
  }

  // execute_reader returns [[name, image], ...] for the parts present:
  // "data" first, then optionally "dark" and "flat".
  auto parts = res.toList();
  for (int i = 0; i < parts.length(); ++i) {
    auto part = parts[i].toList();
    if (part.length() != 2) {
      continue;
    }
    const QString name = part[0].toString();
    auto image = toImage(part[1]);
    if (name == QLatin1String("data")) {
      result.data = image;
    } else if (name == QLatin1String("dark")) {
      result.dark = image;
    } else if (name == QLatin1String("flat")) {
      result.flat = image;
    }
  }
  if (result.data && result.data->GetNumberOfPoints() <= 1) {
    qCritical() << "The file didn't contain any suitable volumetric data";
    result.data = nullptr;
  }
  return result;
}

PythonReaderFactory::PythonReaderFactory(QString description,
                                         QStringList extensions,
                                         Python::Object cls)
  : m_description(description), m_extensions(extensions), m_class(cls)
{
}

PythonReaderFactory::PythonReaderFactory(QString description,
                                         QStringList extensions)
  : m_description(description), m_extensions(extensions)
{
}

QString PythonReaderFactory::getDescription() const
{
  return m_description;
}

QStringList PythonReaderFactory::getExtensions() const
{
  return m_extensions;
}

QString PythonReaderFactory::getFileDialogFilter() const
{
  QStringList filterExt;
  foreach (auto ext, m_extensions) {
    filterExt << QString("*.%1").arg(ext);
  }
  QString filter =
    QString("%1 (%2)").arg(m_description).arg(filterExt.join(QString(" ")));

  return filter;
}

PythonReader PythonReaderFactory::createReader() const
{
  if (!m_class.isValid()) {
    return PythonReader();
  }

  Python python;
  auto module = python.import("tomviz.io._internal");
  if (!module.isValid()) {
    qCritical() << "Failed to import tomviz.io._internal module.";
    return Python::Object();
  }
  auto factory = module.findFunction("create_reader_instance");
  if (!factory.isValid()) {
    qCritical()
      << "Failed to import tomviz.io._internal.create_reader_instance";
    return Python::Object();
  }

  Python::Tuple args(1);
  Python::Object argClass = m_class;
  args.set(0, argClass);
  auto res = factory.call(args);
  if (!res.isValid()) {
    qCritical("Error calling create_reader_instance.");
    return Python::Object();
  }
  return PythonReader(res);
}
}
