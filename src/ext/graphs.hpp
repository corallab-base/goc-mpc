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
	std::vector<int> add_nodes(int n);
	void remove_node(int u);

	void add_edge(int u, int v, const LabelT &label);
	bool remove_edge(int u, int v);

        int num_nodes() const;
	bool directed() const;
	bool alive(int u) const;

	const std::vector<Edge>& neighbors(int u) const;
	std::vector<Edge>& neighbors(int u);

	// In-degree
	std::vector<int> in_degrees() const {
		// Compute (in-)degree among alive nodes.
		int n = num_nodes();
		std::vector<int> indeg(n, 0);
		for (int u = 0; u < n; ++u) {
			if (!_alive[u]) continue;
			for (const auto& e : _adj[u]) {
				const int v = e.to;
				if (v < n && _alive[v]) ++indeg[v];
			}
		}
		return indeg;
	}

	// ------------------- Edge iteration API -------------------

	struct EdgeRef {
		int u;            // source node
		const Edge* e;    // pointer into _adj[u]
	};

	class EdgeIter {
	public:
		EdgeIter(const Graph* g, int u, bool unique_undirected)
			: _g(g), _u(u), _unique(unique_undirected)
			{
				if (_g && _u < _g->num_nodes()) {
					_it  = _g->_adj[_u].begin();
					_end = _g->_adj[_u].end();
					advance_to_valid();
				}
			}

		EdgeRef operator*() const { return { _u, &(*_it) }; }
		const EdgeRef* operator->() const { _ref_cache = { _u, &(*_it) }; return &_ref_cache; }

		EdgeIter& operator++() {
			if (!_g) return *this;
			if (_u < _g->num_nodes() && _it != _end) {
				++_it;
			}
			advance_to_valid();
			return *this;
		}

		bool operator==(const EdgeIter& o) const {
			const int n  = _g  ? _g->num_nodes()  : 0;
			const int on = o._g ? o._g->num_nodes() : 0;
			if ((!_g || _u >= n) && (!o._g || o._u >= on)) return true;
			return _g == o._g && _u == o._u && _it == o._it && _unique == o._unique;
		}
		bool operator!=(const EdgeIter& o) const { return !(*this == o); }

	private:
		void advance_to_valid() {
			if (!_g) return;
			const int n = _g->num_nodes();
			while (_u < n) {
				if (!_g->_alive[_u]) {
					++_u;
					if (_u < n) { _it = _g->_adj[_u].begin(); _end = _g->_adj[_u].end(); }
					continue;
				}
				while (_it != _end) {
					const int v = _it->to;
					if (v < 0 || v >= n || !_g->_alive[v]) { ++_it; continue; }
					if (_unique && !_g->_directed && v < _u) { ++_it; continue; }
					return; // found a valid edge
				}
				++_u;
				if (_u < n) { _it = _g->_adj[_u].begin(); _end = _g->_adj[_u].end(); }
			}
			_it  = typename std::vector<Edge>::const_iterator{};
			_end = typename std::vector<Edge>::const_iterator{};
		}

		const Graph* _g = nullptr;
		int _u = 0;
		bool _unique = false;
		typename std::vector<Edge>::const_iterator _it{};
		typename std::vector<Edge>::const_iterator _end{};
		mutable EdgeRef _ref_cache{ -1, nullptr };
	};

	struct EdgeRange {
		EdgeIter _begin;
		EdgeIter _end;
		EdgeIter begin() const { return _begin; }
		EdgeIter end()   const { return _end; }
	};

	EdgeRange edges(bool unique_undirected = false) const {
		return EdgeRange{
			EdgeIter(this, 0, unique_undirected),
			EdgeIter(this, num_nodes(), unique_undirected)
		};
	}


	std::vector<std::pair<int,int>> incoming_cut_edges(const std::vector<int>& S) const;

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

	// --- incoming_neighbors ---
	// Add these inside InducedSubgraphView<LabelT> public section:

	// (source, e) pair for incoming neighbors where e->to == target u
	struct IncomingNeighborRef { int from; const Edge* e; };

	class IncomingNeighborIter {
	public:
		IncomingNeighborIter(const GraphT* g,
				     const std::vector<bool>* mask,
				     const std::vector<int>* node_list,
				     int target_u,
				     int node_idx)
			: _g(g), _mask(mask), _node_list(node_list),
			  _target(target_u), _node_idx(node_idx) {
			if (_g && _mask && _node_list) {
				advance_to_valid();
			} else {
				_at_end = true;
			}
		}

		IncomingNeighborRef operator*() const { return {_cur_from, &(*_it)}; }

		IncomingNeighborIter& operator++() {
			if (_at_end) return *this;
			// advance within current source's adjacency
			++_it;
			advance_to_valid();
			return *this;
		}

		bool operator==(const IncomingNeighborIter& o) const {
			if (_at_end && o._at_end) return true;
			return _g == o._g
				&& _mask == o._mask
				&& _node_list == o._node_list
				&& _target == o._target
				&& _node_idx == o._node_idx
				&& (_at_end == o._at_end)
				&& (_at_end || _it == o._it);
		}
		bool operator!=(const IncomingNeighborIter& o) const { return !(*this == o); }

	private:
		void advance_to_valid() {
			const int n = _g->num_nodes();
			while (true) {
				if (_node_idx >= static_cast<int>(_node_list->size())) {
					_at_end = true;
					return;
				}
				// move to next source if needed
				if (_it == _end) {
					// advance to next source node in the view
					while (_node_idx < static_cast<int>(_node_list->size())) {
						const int from = (*_node_list)[_node_idx];
						++_node_idx;
						if (from < 0 || from >= n || !_g->alive(from) || !(*_mask)[from]) continue;

						_cur_from = from;
						_it  = _g->neighbors(from).begin();
						_end = _g->neighbors(from).end();
						break;
					}
					// if we reached end of node list, terminate
					if (_it == typename std::vector<Edge>::const_iterator{} &&
					    _node_idx >= static_cast<int>(_node_list->size())) {
						_at_end = true;
						return;
					}
				}

				// scan this source's adjacency to find an edge landing on _target
				while (_it != _end) {
					const int v = _it->to;
					if (v == _target) {
						// (Optionally) ensure target is alive/in view; caller usually checks contains_node(u)
						if (v >= 0 && v < n && _g->alive(v) && (*_mask)[v]) {
							return; // valid incoming edge found
						}
					}
					++_it;
				}
				// loop back to pick next source
			}
		}

		const GraphT* _g = nullptr;
		const std::vector<bool>* _mask = nullptr;
		const std::vector<int>* _node_list = nullptr;

		int _target = -1;
		int _node_idx = 0;
		int _cur_from = -1;

		typename std::vector<Edge>::const_iterator _it{};
		typename std::vector<Edge>::const_iterator _end{};
		bool _at_end = false;
	};

	struct IncomingNeighborRange {
		IncomingNeighborIter _begin;
		IncomingNeighborIter _end;
		IncomingNeighborIter begin() const { return _begin; }
		IncomingNeighborIter end()   const { return _end; }
	};

	// Returns a range of incoming edges to u (from nodes in the view).
	IncomingNeighborRange incoming_neighbors(int u) const {
		if (!contains_node(u)) {
			// empty range
			return IncomingNeighborRange{
				IncomingNeighborIter(nullptr, nullptr, nullptr, -1, 0),
				IncomingNeighborIter(nullptr, nullptr, nullptr, -1, 0)
			};
		}
		return IncomingNeighborRange{
			IncomingNeighborIter(_g, &_mask, &_node_list, u, /*node_idx=*/0),
			IncomingNeighborIter(_g, &_mask, &_node_list, u, static_cast<int>(_node_list.size()))
		};
	}

	// --------


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


	struct TopoLayersResult {
		std::vector<int> order;                // full topological order
		std::vector<std::vector<int>> levels;  // optional: levelized order
		std::map<int, int> node_to_level;
	};

	TopoLayersResult topological_layer_cut_visit(std::function<void(int, int)> cb) const {
		const int n = _g->num_nodes();
		std::vector<std::optional<int>> parent(n, std::nullopt);

		TopoLayersResult res;

		if (_node_list.empty()) return res;

		// In-degree restricted to the view.
		std::vector<int> indeg(n, 0);
		for (int u : _node_list) {
			if (!contains_node(u)) continue;
			for (const auto& e : neighbors(u)) {
				++indeg[e.to]; // neighbors(u) already filtered to nodes in the view
			}
		}

		std::queue<int> q;
		for (int u : _node_list) {
			if (!contains_node(u)) continue;
			if (indeg[u] == 0) {
				q.push(u);
			}
		}

		const int sg_n = _node_list.size();

		res.order.reserve(sg_n);

		while (!q.empty()) {
			int level_sz = (int)q.size();
			std::vector<int> level;
			level.reserve(level_sz);

			// while processing the current level
			while (level_sz--) {
				int u = q.front(); q.pop();
				res.order.push_back(u);
				level.push_back(u);
				for (const auto& e : neighbors(u)) {
					const int v = e.to;
					if (--indeg[v] == 0) {
						q.push(v);
					}
					if (indeg[v] <= 0) {
						cb(u, v);
					}
				}
			}
			res.levels.push_back(std::move(level));
		}

		if ((int)res.order.size() != sg_n)
			throw std::runtime_error("Graph contains a cycle; no topological order.");

		return res;
	}

	TopoLayersResult topological_layer_cut_snapshot(std::function<void(int,int,int)> cb) const {
		const int n = _g->num_nodes();
		TopoLayersResult res;
		if (_node_list.empty()) return res;

		std::vector<int> indeg(n, 0);
		for (int u : _node_list) {
			if (!contains_node(u)) continue;
			for (const auto& e : neighbors(u)) ++indeg[e.to];
		}

		std::queue<int> q;
		for (int u : _node_list) {
			if (!contains_node(u)) continue;
			if (indeg[u] == 0) q.push(u);
		}

		const int sg_n = (int)_node_list.size();
		res.order.reserve(sg_n);

		std::vector<char> processed(n, 0);

		// Accumulate new cut edges contributed by this level
		std::map<int, std::vector<int>> cut_edges_per_v;
		int level_k = 0;

		while (!q.empty()) {
			int level_sz = (int)q.size();
			std::vector<int> level;
			level.reserve(level_sz);

			while (level_sz--) {
				int u = q.front(); q.pop();
				processed[u] = 1;
				res.order.push_back(u);
				level.push_back(u);
				res.node_to_level[u] = level_k;

				// after processing u, remove any existing
				// cut_edges that were coming into it.
				cut_edges_per_v.erase(u);

				// std::cout << "processed " << u << " so removing its incoming edges" << std::endl;

				for (const auto& e : neighbors(u)) {
					int v = e.to;
					if (!processed[v]) {
						// (u,v) now crosses the cut; add to the cut structure
						// std::cout << "adding edge (" << u << ", " << v << ")" << std::endl;
						// std::cout << "before cut_edges_per_v[v].size() = " << cut_edges_per_v[v].size() << std::endl;
						cut_edges_per_v[v].push_back(u);
						// std::cout << "after cut_edges_per_v[v].size() = " << cut_edges_per_v[v].size() << std::endl;
					}
					if (--indeg[v] == 0) q.push(v);
				}
			}

			// Emit the snapshot of the entire cut after finishing this level:
			for (auto& [v, us] : cut_edges_per_v) {
				for (int u : us) {
					cb(level_k, u, v);
				}
			}

			res.levels.push_back(std::move(level));
			level_k++;
		}

		if ((int)res.order.size() != sg_n)
			throw std::runtime_error("Graph contains a cycle; no topological order.");

		return res;
	}

private:
	const GraphT* _g;               // parent graph (not owning)
	std::vector<bool>   _mask;      // membership mask by node id
	std::vector<int> _node_list; // compact node list (for fast outer loops)
	std::map<int, int> _subgraph_id_map;
};
