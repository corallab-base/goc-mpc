#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <stdexcept>
#include <functional>
#include <optional>
#include <utility>
#include <algorithm>


template <typename LabelT>
class Graph {
public:
	struct Edge {
		size_t to;
		LabelT label;
	};

	explicit Graph(bool directed = true);

	static Graph WithNodes(size_t n, bool directed = true);

	size_t add_node();
	void add_nodes(size_t n);
	void remove_node(size_t u);

	void add_edge(size_t u, size_t v, const LabelT &label);
	bool remove_edge(size_t u, size_t v);

	size_t num_nodes() const;
	bool directed() const;
	bool alive(size_t u) const;

	const std::vector<Edge>& neighbors(size_t u) const;
	std::vector<Edge>& neighbors(size_t u);

	struct BFSResult { std::vector<std::optional<size_t>> parent; std::vector<int> dist; };
	BFSResult bfs(size_t s) const;

	// DFS from a specific start node; calls cb(u, parent) on discovery (preorder).
	// Skips inactive nodes. parent = std::nullopt at root.
	void dfs_visit(size_t s,
		       const std::function<void(size_t, std::optional<size_t>)>& cb) const;

	// DFS starting from all "source" nodes (zero in-degree among alive nodes).
	// For undirected graphs, sources are the zero-degree nodes.
	// Calls cb(u, parent) for each discovered node exactly once.
	void dfs_visit_from_sources(
		const std::function<void(size_t, std::optional<size_t>)>& cb) const;

	std::vector<size_t> dfs(size_t s) const;

	struct SSSPResult { std::vector<double> dist; std::vector<std::optional<size_t>> parent; };
	SSSPResult dijkstra(size_t s, std::function<double(const LabelT&)> weight_fn) const;

	std::vector<size_t> topological_sort() const;

private:
	void check_node(size_t u) const;

	bool _directed;
	size_t _nodes_alive = 0;
	std::vector<std::vector<Edge>> _adj;
	std::vector<bool> _alive;
};

/// Read-only, zero-copy induced subgraph view over Graph<LabelT>.
/// Lifetime note: invalidated by structural mutations of the parent Graph
/// (adding/removing nodes/edges). Safe against non-structural const reads.
template <typename LabelT>
class InducedSubgraphView {
public:
	using GraphT = Graph<LabelT>;
	using Edge   = typename GraphT::Edge;

	/// Construct from parent graph and a list of nodes to keep.
	InducedSubgraphView(const GraphT& g, const std::vector<size_t>& nodes);

	size_t num_nodes() const;

	/// True if u is in the view and still alive in the parent.
	bool contains_node(size_t u) const;

	/// Unique integer from 0 - num_nodes for vertex u in subgraph.
	size_t subgraph_id(size_t u) const;

	/// Out-degree in the subgraph (directed graphs). Returns 0 if u not in view.
	size_t degree(size_t u) const;

	/// Forward iterator filtering a parent's neighbor list on the mask.
	class NeighborIter {
	public:
		using InnerIter = typename std::vector<Edge>::const_iterator;

		NeighborIter(const GraphT* g,
			     const std::vector<bool>* mask,
			     size_t u,
			     InnerIter it,
			     InnerIter end);

		const Edge& operator*() const;
		const Edge* operator->() const;
		NeighborIter& operator++();
		bool operator==(const NeighborIter& o) const;
		bool operator!=(const NeighborIter& o) const;

	private:
		void advance_to_valid();

		const GraphT* _g = nullptr;
		const std::vector<bool>* _mask = nullptr;
		size_t _u = 0;
		InnerIter _it{};
		InnerIter _end{};
	};

	struct NeighborRange {
		NeighborIter _begin;
		NeighborIter _end;
		NeighborIter begin() const { return _begin; }
		NeighborIter end()   const { return _end; }
	};

	/// Subgraph-filtered neighbors of u. Empty range if u not contained.
	NeighborRange neighbors(size_t u) const;

	/// Nodes present in the subgraph (compact list).
	const std::vector<size_t>& nodes() const { return _node_list; }

	/// (u, e) pair when iterating edges() without copying edges.
	struct EdgeRef { size_t u; const Edge* e; };

	/// Iterator over all edges in the subgraph (outer loop over nodes in view).
	class EdgeIter {
	public:
		EdgeIter(const GraphT* g,
			 const std::vector<bool>* mask,
			 const std::vector<size_t>* node_list,
			 size_t node_idx);

		EdgeRef operator*() const;
		EdgeIter& operator++();
		bool operator==(const EdgeIter& o) const;
		bool operator!=(const EdgeIter& o) const;

	private:
		void advance_node();
		void advance_edge();

		const GraphT* _g = nullptr;
		const std::vector<bool>* _mask = nullptr;
		const std::vector<size_t>* _node_list = nullptr;
		size_t _node_idx = 0;

		typename std::vector<Edge>::const_iterator _it{};
		typename std::vector<Edge>::const_iterator _end_it{};
		bool _at_end = false;
	};

	struct EdgeRange {
		EdgeIter _begin;
		EdgeIter _end;
		EdgeIter begin() const { return _begin; }
		EdgeIter end()   const { return _end; }
	};

	/// Iterate all edges (u -> e->to) in the induced view.
	EdgeRange edges() const;

	/// Access the parent graph (const).
	const GraphT& parent() const { return *_g; }


	// DFS within the view from a specific start node (must be contained).
	// Calls cb(u, parent) on discovery. Skips nodes not in the view.
	void dfs_visit(size_t s,
		       const std::function<void(size_t, std::optional<size_t>)>& cb) const;

	std::vector<size_t> sources() const;

	// DFS within the view starting from all view-sources (zero in-degree inside the view).
	void dfs_visit_from_sources(
		const std::function<void(size_t, std::optional<size_t>)>& cb) const;

private:
	const GraphT* _g;               // parent graph (not owning)
	std::vector<bool>   _mask;      // membership mask by node id
	std::vector<size_t> _node_list; // compact node list (for fast outer loops)
	std::map<size_t, size_t> _subgraph_id_map;
};
