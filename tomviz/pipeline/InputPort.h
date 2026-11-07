/* This source file is part of the Tomviz project, https://tomviz.org/.
   It is released under the 3-Clause BSD License, see "LICENSE". */

#ifndef tomvizPipelineInputPort_h
#define tomvizPipelineInputPort_h

#include "Port.h"
#include "PortData.h"
#include "PortType.h"

#include <memory>

namespace tomviz {
namespace pipeline {

class Link;
class OutputPort;

class InputPort : public Port
{
  Q_OBJECT

public:
  InputPort(const QString& name, PortTypes acceptedTypes,
            QObject* parent = nullptr);
  ~InputPort() override = default;

  PortTypes acceptedTypes() const;
  void setAcceptedTypes(PortTypes types);
  bool canConnectTo(const OutputPort* output) const;
  Link* link() const;
  PortData data() const;
  bool hasData() const;
  bool isStale() const;

  /// An optional input may stay unconnected: the node still runs and
  /// receives no data for it (Python operators get their default, None).
  bool isOptional() const;
  void setOptional(bool optional);

  /// Delivery channel for the upstream payload during executor-driven
  /// runs. The executor calls setHandle() before invoking the consumer
  /// node and clearHandle() after. Sinks that need their input alive
  /// past end-of-plan call handle() to grab the shared_ptr and stash
  /// it in a member. Read paths (data()/hasData()) deliberately ignore
  /// this — they observe via the upstream output port so UI callers
  /// outside of execute() see the same picture.
  const std::shared_ptr<PortData>& handle() const;
  void setHandle(std::shared_ptr<PortData> handle);
  void clearHandle();

private:
  friend class Link;
  void setLink(Link* link);

  PortTypes m_acceptedTypes;
  Link* m_link = nullptr;
  bool m_optional = false;
  std::shared_ptr<PortData> m_handle;
};

} // namespace pipeline
} // namespace tomviz

#endif
