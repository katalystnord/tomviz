/* This source file is part of the Tomviz project, https://tomviz.org/.
   It is released under the 3-Clause BSD License, see "LICENSE". */

#include "GatedEditorWidget.h"

#include "InputPort.h"
#include "InputsNotReadyWidget.h"
#include "Link.h"
#include "Node.h"
#include "OutputPort.h"
#include "Pipeline.h"

#include <QVBoxLayout>

namespace tomviz {
namespace pipeline {

GatedEditorWidget::GatedEditorWidget(Node* node, Pipeline* pipeline,
                                     EditorFactory factory, QWidget* parent)
  : EditNodeWidget(parent), m_node(node), m_pipeline(pipeline),
    m_factory(std::move(factory))
{
  m_layout = new QVBoxLayout(this);
  m_layout->setContentsMargins(0, 0, 0, 0);

  if (inputsReady()) {
    buildInner();
  } else {
    m_notReadyWidget = new InputsNotReadyWidget(this);
    connect(m_notReadyWidget, &InputsNotReadyWidget::runRequested,
            this, &GatedEditorWidget::onRunRequested);
    m_notReadyWidget->setRunEnabled(!m_pipeline->isExecuting());
    m_layout->addWidget(m_notReadyWidget, 1);

    connect(m_pipeline, &Pipeline::executionStarted, this, [this]() {
      if (m_notReadyWidget) {
        m_notReadyWidget->setRunEnabled(false);
      }
    });
    connect(m_pipeline, &Pipeline::executionFinished,
            this, &GatedEditorWidget::onExecutionFinished);
  }
}

bool GatedEditorWidget::canApply() const
{
  return m_inner != nullptr;
}

void GatedEditorWidget::applyChangesToOperator()
{
  if (m_inner) {
    m_inner->applyChangesToOperator();
  }
}

bool GatedEditorWidget::inputsReady() const
{
  if (!m_node) {
    return false;
  }
  for (auto* input : m_node->inputPorts()) {
    if (!input->link() && input->isOptional()) {
      continue;
    }
    if (!input->link() || input->isStale() || !input->hasData()) {
      return false;
    }
    if (input->link()->from()->dataLocation() != DataLocation::InMemory) {
      return false;
    }
  }
  return true;
}

void GatedEditorWidget::onRunRequested()
{
  if (!m_node) {
    return;
  }
  m_pipeline->executeUpstreamOf(m_node);
}

void GatedEditorWidget::onExecutionFinished()
{
  if (!m_notReadyWidget || !m_node) {
    return;
  }

  // Pin freshest data into memory so inputsReady() flips true and the
  // inner editor can read via data() without I/O.
  for (auto* input : m_node->inputPorts()) {
    auto* link = input->link();
    if (!link || !link->from()) {
      continue;
    }
    if (auto handle = link->from()->materialize()) {
      m_inputPins.append(handle);
    }
  }

  if (!inputsReady()) {
    m_notReadyWidget->setRunEnabled(true);
    return;
  }

  // Build the inner editor first. The factory can legitimately return null if
  // the upstream data isn't actually usable yet (e.g. this finished signal was
  // delivered re-entrantly mid-execution); in that case stay in the not-ready
  // state rather than tearing down the widget and showing a blank editor.
  if (!buildInner()) {
    m_notReadyWidget->setRunEnabled(true);
    return;
  }

  m_layout->removeWidget(m_notReadyWidget);
  m_notReadyWidget->deleteLater();
  m_notReadyWidget = nullptr;
}

bool GatedEditorWidget::buildInner()
{
  // The factory closure captures the target node by raw pointer, so never
  // invoke it once the node is gone - guard here so buildInner() is safe
  // regardless of which caller reaches it.
  if (!m_node) {
    return false;
  }
  m_inner = m_factory(this);
  if (m_inner) {
    m_layout->addWidget(m_inner, 1);
  }
  emit canApplyChanged();
  return m_inner != nullptr;
}

} // namespace pipeline
} // namespace tomviz
