/* This source file is part of the Tomviz project, https://tomviz.org/.
   It is released under the 3-Clause BSD License, see "LICENSE". */

#ifndef tomvizPipelinePipelineUtils_h
#define tomvizPipelinePipelineUtils_h

#include <QSet>

namespace tomviz {
namespace pipeline {

class InputPort;
class LegacyModuleSink;
class Link;
class Node;
class OutputPort;
class Pipeline;

/// Find the tip output port of the branch containing the given node.
/// Walks downstream from the node through TransformNodes to find the end
/// of that specific branch. If the node is a SinkNode, walks upstream first
/// to find the feeding source/transform, then walks downstream from there.
OutputPort* findBranchTip(Node* node);

/// Find the tip output port using contextNode to select the right branch.
/// If contextNode is null, falls back to the first source in the pipeline.
OutputPort* findTipOutputPort(
  Pipeline* pipeline, Node* contextNode);

/// The output port of the source whose data reaches @a port, found by
/// walking upstream along each node's first linked input: @a port itself
/// when its node is a source, nullptr when no source feeds it.
OutputPort* feedingSourcePort(OutputPort* port);

/// The same for a whole node, whether or not it has outputs: a source's
/// first output, or the source feeding the node's first linked input.
OutputPort* feedingSourcePort(Node* node);

/// The output port a new sink's @a input should be linked to so the sink
/// hangs off a SinkGroupNode on @a targetPort: the port itself when it
/// already belongs to a group, the matching passthrough of a compatible
/// group already linked to it, or otherwise a new group created on it.
/// Returns nullptr when @a input does not accept @a targetPort's type.
OutputPort* sinkAttachPort(Pipeline* pipeline, OutputPort* targetPort,
                           InputPort* input);

/// The port that actually carries @a port's data: a SinkGroupNode
/// passthrough is followed up to the port feeding the group. @a skip is
/// a link to treat as already gone (Pipeline::linkRemoved fires while
/// the link is still attached).
OutputPort* dataPort(OutputPort* port, const Link* skip = nullptr);

/// Every display fed by @a port, directly or through Visualizations
/// groups, ignoring the link @a skip.
QSet<LegacyModuleSink*> displaysOf(OutputPort* port,
                                   const Link* skip = nullptr);

/// Called when the user wires @a link into an operator's extra input (a
/// dataset parameter such as a flat field, not its first input): if the
/// dataset feeds nothing but such inputs, its own displays are hidden,
/// since they are not what the user is looking at and are usually drawn
/// at a different scale. Returns how many displays were hidden.
int hideDisplaysOfAuxiliaryInput(Link* link);

} // namespace pipeline
} // namespace tomviz

#endif
