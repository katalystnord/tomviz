/* This source file is part of the Tomviz project, https://tomviz.org/.
   It is released under the 3-Clause BSD License, see "LICENSE". */

#include "InputPort.h"

#include "Link.h"
#include "OutputPort.h"

namespace tomviz {
namespace pipeline {

InputPort::InputPort(const QString& name, PortTypes acceptedTypes,
                     QObject* parent)
  : Port(name, parent), m_acceptedTypes(acceptedTypes)
{}

PortTypes InputPort::acceptedTypes() const
{
  return m_acceptedTypes;
}

void InputPort::setAcceptedTypes(PortTypes types)
{
  m_acceptedTypes = types;
}

bool InputPort::canConnectTo(const OutputPort* output) const
{
  if (!output) {
    return false;
  }
  return isPortTypeCompatible(output->type(), m_acceptedTypes);
}

Link* InputPort::link() const
{
  return m_link;
}

PortData InputPort::data() const
{
  if (m_link) {
    return m_link->from()->data();
  }
  return PortData();
}

bool InputPort::hasData() const
{
  if (m_link) {
    return m_link->from()->hasData();
  }
  return false;
}

bool InputPort::isStale() const
{
  if (m_link) {
    return m_link->from()->isStale();
  }
  return false;
}

const std::shared_ptr<PortData>& InputPort::handle() const
{
  return m_handle;
}

void InputPort::setHandle(std::shared_ptr<PortData> handle)
{
  m_handle = std::move(handle);
}

void InputPort::clearHandle()
{
  m_handle.reset();
}

void InputPort::setLink(Link* link)
{
  m_link = link;
  emit connectionChanged();
}

bool InputPort::isOptional() const
{
  return m_optional;
}

void InputPort::setOptional(bool optional)
{
  m_optional = optional;
}

} // namespace pipeline
} // namespace tomviz
