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
	        int to;
		LabelT label;
	};

	explicit Graph(bool directed = true);

	static Graph WithNodes(int n, bool directed = true);

        int add_node();
	void add_nodes(int n);
	void remove_node(int u);

	void add_edge(int u, int v, const LabelT &label);
	bool remove_edge(int u, int v);

        int num_nodes() const;
	bool directed() const;
	bool alive(int u) const;

	const std::vector<Edge>& neighbors(int u) const;
	std::vector<Edge>& neighbors(int u);

	struct BFSResult { std::vector<std::optional<int>> parent; std::vector<int> dist; };
	BFSResult bfs(int s) const;

	std::vector<std::optional<int>> bfs_visit_from_sources(
		const std::function<void(int, std::optional<int>)>& cb,
		const std::function<void(int, int)>& non_tree_cb) const;

	// DFS from a specific start node; calls cb(u, parent) on discovery (preorder).
	// Skips inactive nodes. parent = std::nullopt at root.
	void dfs_visit(int s,
		       const std::function<void(int, std::optional<int>)>& cb) const;

	// DFS starting from all "source" nodes (zero in-degree among alive nodes).
	// For undirected graphs, sources are the zero-degree nodes.
	// Calls cb(u, parent) for each discovered node exactly once.
	void dfs_visit_from_sources(
		const std::function<void(int, std::optional<int>)>& cb) const;

	std::vector<int> dfs(int s) const;

	struct SSSPResult { std::vector<double> dist; std::vector<std::optional<int>> parent; };
	SSSPResult dijkstra(int s, std::function<double(const LabelT&)> weight_fn) const;

	std::vector<int> topological_sort() const;

private:
	void check_node(int u) const;

	bool _directed;
        int _nodes_alive = 0;
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
	InducedSubgraphView(const GraphT& g, const std::vector<int>& nodes);

        int num_nodes() const;

	/// True if u is in the view and still alive in the parent.
	bool contains_node(int u) const;

	/// Unique integer from 0 - num_nodes for vertex u in subgraph.
        int subgraph_id(int u) const;

	/// Out-degree in the subgraph (directed graphs). Returns 0 if u not in view.
        int degree(int u) const;

	/// Forward iterator filtering a parent's neighbor list on the mask.
	class NeighborIter {
	public:
		using InnerIter = typename std::vector<Edge>::const_iterator;

		NeighborIter(const GraphT* g,
			     const std::vector<bool>* mask,
			     int u,
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
	        int _u = 0;
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
	NeighborRange neighbors(int u) const;

	/// Nodes present in the subgraph (compact list).
	const std::vector<int>& nodes() const { return _node_list; }

	/// (u, e) pair when iterating edges() without copying edges.
	struct EdgeRef { int u; const Edge* e; };

	/// Iterator over all edges in the subgraph (outer loop over nodes in view).
	class EdgeIter {
	public:
		EdgeIter(const GraphT* g,
			 const std::vector<bool>* mask,
			 const std::vector<int>* node_list,
			 int node_idx);

		EdgeRef operator*() const;
		EdgeIter& operator++();
		bool operator==(const EdgeIter& o) const;
		bool operator!=(const EdgeIter& o) const;

	private:
		void advance_node();
		void advance_edge();

		const GraphT* _g = nullptr;
		const std::vector<bool>* _mask = nullptr;
		const std::vector<int>* _node_list = nullptr;
	        int _node_idx = 0;

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
	void dfs_visit(int s,
		       const std::function<void(int, std::optional<int>)>& cb) const;

	std::vector<int> sources() const;

	// DFS within the view starting from all view-sources (zero in-degree inside the view).
	void dfs_visit_from_sources(
		const std::function<void(int, std::optional<int>)>& cb) const;

	// BFS

	std::vector<std::optional<int>> bfs_visit_from_sources(
		const std::function<void(int, int, std::optional<int>)>& cb,
		const std::function<void(int, int, int, int)>& non_tree_cb) const;

private:
	const GraphT* _g;               // parent graph (not owning)
	std::vector<bool>   _mask;      // membership mask by node id
	std::vector<int> _node_list; // compact node list (for fast outer loops)
	std::map<int, int> _subgraph_id_map;
};
