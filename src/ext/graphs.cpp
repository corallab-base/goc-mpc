#include "graphs.hpp"

namespace py = pybind11;


template <typename LabelT>
Graph<LabelT>::Graph(bool directed) : _directed(directed) {}

template <typename LabelT>
Graph<LabelT> Graph<LabelT>::WithNodes(size_t n, bool directed) {
	Graph g(directed);
	g.add_nodes(n);
	return g;
}

template <typename LabelT>
size_t Graph<LabelT>::add_node() {
	size_t id = _nodes_alive;
	++_nodes_alive;
	_adj.emplace_back();
	_alive.push_back(true);
	return id;
}

template <typename LabelT>
void Graph<LabelT>::add_nodes(size_t n) {
	for (size_t i = 0; i < n; ++i) add_node();
}

template <typename LabelT>
void Graph<LabelT>::remove_node(size_t u) {
	check_node(u);
	if (!_alive[u]) return;
	_alive[u] = false;
	_adj[u].clear();
	if (!_directed) {
		for (auto &nbrs : _adj) {
			nbrs.erase(std::remove_if(nbrs.begin(), nbrs.end(), [&](const Edge &e){return e.to==u;}), nbrs.end());
		}
	}
}

template <typename LabelT>
void Graph<LabelT>::add_edge(size_t u, size_t v, const LabelT &label) {
	check_node(u); check_node(v);
	if (!_alive[u] || !_alive[v]) throw std::invalid_argument("add_edge: node is removed");
	_adj[u].push_back(Edge{v, label});
	if (!_directed && u != v) {
		_adj[v].push_back(Edge{u, label});
	}
}

template <typename LabelT>
bool Graph<LabelT>::remove_edge(size_t u, size_t v) {
	check_node(u); check_node(v);
	auto &nbrs = _adj[u];
	auto it = std::find_if(nbrs.begin(), nbrs.end(), [&](const Edge &e){ return e.to==v; });
	bool removed = false;
	if (it != nbrs.end()) { nbrs.erase(it); removed = true; }
	if (!_directed) {
		auto &nbrs2 = _adj[v];
		auto it2 = std::find_if(nbrs2.begin(), nbrs2.end(), [&](const Edge &e){ return e.to==u; });
		if (it2 != nbrs2.end()) { nbrs2.erase(it2); removed = true; }
	}
	return removed;
}

template <typename LabelT>
size_t Graph<LabelT>::num_nodes() const { return _adj.size(); }

template <typename LabelT>
bool Graph<LabelT>::directed() const { return _directed; }

template <typename LabelT>
bool Graph<LabelT>::alive(size_t u) const { check_node(u); return _alive[u]; }

template <typename LabelT>
const std::vector<typename Graph<LabelT>::Edge>& Graph<LabelT>::neighbors(size_t u) const { check_node(u); return _adj[u]; }

template <typename LabelT>
std::vector<typename Graph<LabelT>::Edge>& Graph<LabelT>::neighbors(size_t u) { check_node(u); return _adj[u]; }

template <typename LabelT>
typename Graph<LabelT>::BFSResult Graph<LabelT>::bfs(size_t s) const {
	check_node(s);
	const int INF = std::numeric_limits<int>::max();
	std::vector<int> dist(num_nodes(), INF);
	std::vector<std::optional<size_t>> parent(num_nodes(), std::nullopt);
	if (!_alive[s]) return {std::move(parent), std::move(dist)};

	std::queue<size_t> q; q.push(s); dist[s]=0; parent[s]=s;
	while(!q.empty()){
		size_t u=q.front(); q.pop();
		for(const auto& e: _adj[u]){
			size_t v=e.to; if(!_alive[v]) continue;
			if(dist[v]==INF){ dist[v]=dist[u]+1; parent[v]=u; q.push(v);} }
	}
	return {std::move(parent), std::move(dist)};
}

template <typename LabelT>
std::vector<size_t> Graph<LabelT>::dfs(size_t s) const {
	check_node(s);
	std::vector<size_t> out;
	if (!_alive[s]) return out;
	std::vector<char> vis(num_nodes(), 0);
	std::stack<size_t> st; st.push(s);
	while(!st.empty()){
		size_t u=st.top(); st.pop();
		if(vis[u]||!_alive[u]) continue; vis[u]=1; out.push_back(u);
		const auto &nbrs = _adj[u];
		for(auto it = nbrs.rbegin(); it!=nbrs.rend(); ++it){ st.push(it->to);}
	}
	return out;
}

template <typename LabelT>
void Graph<LabelT>::dfs_visit(
	size_t s,
	const std::function<void(size_t, std::optional<size_t>)>& cb) const
{
	check_node(s);
	if (!_alive[s]) return;

	const size_t n = num_nodes();
	std::vector<char> vis(n, 0);

	using Frame = std::pair<size_t, std::optional<size_t>>; // (u, parent)
	std::vector<Frame> st;
	st.reserve(n);
	st.emplace_back(s, std::nullopt);

	while (!st.empty()) {
		auto [u, parent] = st.back();
		st.pop_back();

		if (u >= n || vis[u] || !_alive[u]) continue;
		vis[u] = 1;

		cb(u, parent);  // preorder

		// Push neighbors in reverse to get a stable left-to-right preorder.
		const auto& nbrs = _adj[u];
		for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it) {
			const size_t v = it->to;
			if (v < n && !vis[v] && _alive[v]) st.emplace_back(v, u);
		}
	}
}

template <typename LabelT>
void Graph<LabelT>::dfs_visit_from_sources(
	const std::function<void(size_t, std::optional<size_t>)>& cb) const
{
	const size_t n = num_nodes();
	if (n == 0) return;

	// Compute (in-)degree among alive nodes.
	std::vector<size_t> indeg(n, 0);
	for (size_t u = 0; u < n; ++u) {
		if (!_alive[u]) continue;
		for (const auto& e : _adj[u]) {
			const size_t v = e.to;
			if (v < n && _alive[v]) ++indeg[v];
		}
	}

	std::vector<char> vis(n, 0);

	auto do_dfs = [&](size_t root) {
		using Frame = std::pair<size_t, std::optional<size_t>>;
		std::vector<Frame> st;
		st.emplace_back(root, std::nullopt);
		while (!st.empty()) {
			auto [u, parent] = st.back();
			st.pop_back();
			if (u >= n || vis[u] || !_alive[u]) continue;
			vis[u] = 1;
			cb(u, parent);
			const auto& nbrs = _adj[u];
			for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it) {
				const size_t v = it->to;
				if (v < n && !vis[v] && _alive[v]) st.emplace_back(v, u);
			}
		}
	};

	// Start from sources; for undirected graphs this means zero-degree nodes.
	for (size_t u = 0; u < n; ++u) {
		if (_alive[u] && indeg[u] == 0 && !vis[u]) do_dfs(u);
	}

	// Optional: if you want to ensure every alive node is visited even when
	// there are cycles / no sources, you could fall back here:
	// for (size_t u = 0; u < n; ++u) if (_alive[u] && !vis[u]) do_dfs(u);
}

template <typename LabelT>
typename Graph<LabelT>::SSSPResult Graph<LabelT>::dijkstra(size_t s, std::function<double(const LabelT&)> weight_fn) const {
	check_node(s);
	const double INF = std::numeric_limits<double>::infinity();
	std::vector<double> dist(num_nodes(), INF);
	std::vector<std::optional<size_t>> parent(num_nodes(), std::nullopt);
	if (!_alive[s]) return {std::move(dist), std::move(parent)};

	using QItem = std::pair<double,size_t>;
	struct Cmp { bool operator()(const QItem& a, const QItem& b) const { return a.first > b.first; } };
	std::priority_queue<QItem, std::vector<QItem>, Cmp> pq;
	dist[s]=0.0; parent[s]=s; pq.emplace(0.0, s);

	while(!pq.empty()){
		auto [du,u]=pq.top(); pq.pop(); if(du!=dist[u]) continue; if(!_alive[u]) continue;
		for(const auto& e: _adj[u]){
			size_t v=e.to; if(!_alive[v]) continue;
			double w = weight_fn(e.label);
			if(w < 0) throw std::invalid_argument("dijkstra: negative edge weight");
			if(dist[v] > du + w){ dist[v] = du + w; parent[v] = u; pq.emplace(dist[v], v);} }
	}
	return {std::move(dist), std::move(parent)};
}

template <typename LabelT>
std::vector<size_t> Graph<LabelT>::topological_sort() const {
	if (!_directed) throw std::logic_error("topological_sort: only for directed graphs");
	const size_t n = num_nodes();
	std::vector<size_t> indeg(n, 0);
	for(size_t u=0; u<n; ++u){ if(!_alive[u]) continue; for(const auto &e: _adj[u]) if(_alive[e.to]) ++indeg[e.to]; }
	std::queue<size_t> q; for(size_t u=0; u<n; ++u) if(_alive[u] && indeg[u]==0) q.push(u);
	std::vector<size_t> order; order.reserve(n);
	while(!q.empty()){
		size_t u=q.front(); q.pop(); order.push_back(u);
		for(const auto &e: _adj[u]){ if(!_alive[e.to]) continue; if(--indeg[e.to]==0) q.push(e.to); }
	}
	size_t alive_count = 0; for(bool a: _alive) if(a) ++alive_count;
	if(order.size() != alive_count)
		throw std::runtime_error("Graph has at least one cycle among alive nodes");
	return order;
}

template <typename LabelT>
void Graph<LabelT>::check_node(size_t u) const {
	if (u >= _adj.size()) throw std::out_of_range("node id out of range");
}

/*
 * InducedSubgraphView
 */

template <typename LabelT>
InducedSubgraphView<LabelT>::InducedSubgraphView(const GraphT& g,
                                                 const std::vector<size_t>& nodes)
	: _g(&g),
	  _mask(g.num_nodes(), false) {
	_node_list.reserve(nodes.size());
	size_t i = 0;
	for (size_t u : nodes) {
		if (u < g.num_nodes() && g.alive(u) && !_mask[u]) {
			_mask[u] = true;
			_subgraph_id_map[u] = i++;
			_node_list.push_back(u);
		}
	}
}

template <typename LabelT>
size_t InducedSubgraphView<LabelT>::num_nodes() const {
	return _node_list.size();
}

template <typename LabelT>
bool InducedSubgraphView<LabelT>::contains_node(size_t u) const {
	return u < _mask.size() && _mask[u] && _g->alive(u);
}

template <typename LabelT>
size_t InducedSubgraphView<LabelT>::subgraph_id(size_t u) const {
	if (!contains_node(u)) return -1;
	return _subgraph_id_map.at(u);
}

template <typename LabelT>
size_t InducedSubgraphView<LabelT>::degree(size_t u) const {
	if (!contains_node(u)) return 0;
	size_t d = 0;
	const auto& nbrs = _g->neighbors(u);
	for (const auto& e : nbrs) if (contains_node(e.to)) ++d;
	return d;
}

// ---- NeighborIter ----------------------------------------------------------

template <typename LabelT>
InducedSubgraphView<LabelT>::NeighborIter::NeighborIter(
	const GraphT* g,
	const std::vector<bool>* mask,
	size_t u,
	InnerIter it,
	InnerIter end)
	: _g(g), _mask(mask), _u(u), _it(it), _end(end)
{
	advance_to_valid();
}

template <typename LabelT>
const typename InducedSubgraphView<LabelT>::Edge&
InducedSubgraphView<LabelT>::NeighborIter::operator*() const {
	return *_it;
}

template <typename LabelT>
const typename InducedSubgraphView<LabelT>::Edge*
InducedSubgraphView<LabelT>::NeighborIter::operator->() const {
	return &*_it;
}

template <typename LabelT>
typename InducedSubgraphView<LabelT>::NeighborIter&
InducedSubgraphView<LabelT>::NeighborIter::operator++() {
	++_it;
	advance_to_valid();
	return *this;
}

template <typename LabelT>
bool InducedSubgraphView<LabelT>::NeighborIter::operator==(
	const NeighborIter& o) const {
	return _it == o._it;
}

template <typename LabelT>
bool InducedSubgraphView<LabelT>::NeighborIter::operator!=(
	const NeighborIter& o) const {
	return !(*this == o);
}

template <typename LabelT>
void InducedSubgraphView<LabelT>::NeighborIter::advance_to_valid() {
	while (_it != _end) {
		const size_t v = _it->to;
		if (v < _mask->size() && (*_mask)[v] && _g->alive(v)) break;
		++_it;
	}
}

// ---- neighbors(u) ----------------------------------------------------------

template <typename LabelT>
typename InducedSubgraphView<LabelT>::NeighborRange
InducedSubgraphView<LabelT>::neighbors(size_t u) const {
	if (!contains_node(u)) {
		// empty range: begin == end
		auto dummy = (_g->num_nodes() ? std::min(u, _g->num_nodes() - 1) : 0);
		const auto& nbrs = _g->neighbors(dummy);
		auto it = nbrs.end();
		return NeighborRange{
			NeighborIter(_g, &_mask, u, it, it),
			NeighborIter(_g, &_mask, u, it, it)
		};
	}
	const auto& nbrs = _g->neighbors(u);
	return NeighborRange{
		NeighborIter(_g, &_mask, u, nbrs.begin(), nbrs.end()),
		NeighborIter(_g, &_mask, u, nbrs.end(),   nbrs.end())
	};
}

// ---- EdgeIter --------------------------------------------------------------

template <typename LabelT>
InducedSubgraphView<LabelT>::EdgeIter::EdgeIter(
	const GraphT* g,
	const std::vector<bool>* mask,
	const std::vector<size_t>* node_list,
	size_t node_idx)
	: _g(g), _mask(mask), _node_list(node_list), _node_idx(node_idx)
{
	advance_node();
}

template <typename LabelT>
typename InducedSubgraphView<LabelT>::EdgeRef
InducedSubgraphView<LabelT>::EdgeIter::operator*() const {
	return EdgeRef{ _node_list->at(_node_idx), &*_it };
}

template <typename LabelT>
typename InducedSubgraphView<LabelT>::EdgeIter&
InducedSubgraphView<LabelT>::EdgeIter::operator++() {
	++_it;
	advance_edge();
	return *this;
}

template <typename LabelT>
bool InducedSubgraphView<LabelT>::EdgeIter::operator==(
	const EdgeIter& o) const {
	if (_at_end && o._at_end) return true;
	return _node_idx == o._node_idx && _it == o._it;
}

template <typename LabelT>
bool InducedSubgraphView<LabelT>::EdgeIter::operator!=(
	const EdgeIter& o) const {
	return !(*this == o);
}

template <typename LabelT>
void InducedSubgraphView<LabelT>::EdgeIter::advance_node() {
	while (_node_idx < _node_list->size()) {
		size_t u = _node_list->at(_node_idx);
		const auto& nbrs = _g->neighbors(u);
		_it = nbrs.begin();
		_end_it = nbrs.end();
		advance_edge();
		if (!_at_end) return;
		++_node_idx;
	}
	_at_end = true;
}

template <typename LabelT>
void InducedSubgraphView<LabelT>::EdgeIter::advance_edge() {
	while (_node_idx < _node_list->size()) {
		while (_it != _end_it) {
			const size_t v = _it->to;
			if (v < _mask->size() && (*_mask)[v] && _g->alive(v)) {
				_at_end = false;
				return;
			}
			++_it;
		}
		// move to next node
		++_node_idx;
		if (_node_idx >= _node_list->size()) {
			_at_end = true;
			return;
		}
		const auto& nbrs = _g->neighbors(_node_list->at(_node_idx));
		_it = nbrs.begin();
		_end_it = nbrs.end();
	}
	_at_end = true;
}

// ---- edges() ---------------------------------------------------------------

template <typename LabelT>
typename InducedSubgraphView<LabelT>::EdgeRange
InducedSubgraphView<LabelT>::edges() const {
	return EdgeRange{
		EdgeIter(_g, &_mask, &_node_list, 0),
		EdgeIter(_g, &_mask, &_node_list, _node_list.size())
	};
}

// DFS

template <typename LabelT>
void InducedSubgraphView<LabelT>::dfs_visit(
	size_t s,
	const std::function<void(size_t, std::optional<size_t>)>& cb) const
{
	if (!contains_node(s)) return;

	const size_t n = _g->num_nodes();
	std::vector<char> vis(n, 0);

	using Frame = std::pair<size_t, std::optional<size_t>>;
	std::vector<Frame> st;
	st.emplace_back(s, std::nullopt);

	while (!st.empty()) {
		auto [u, parent] = st.back(); st.pop_back();
		if (u >= n || vis[u] || !contains_node(u)) continue;
		vis[u] = 1;

		cb(u, parent);

		// Collect filtered neighbors first, then push in reverse for stable order.
		std::vector<size_t> tmp;
		for (const auto& e : neighbors(u)) {
			const size_t v = e.to; // guaranteed in view by neighbors()
			if (v < n && !vis[v] && contains_node(v)) tmp.push_back(v);
		}
		for (auto it = tmp.rbegin(); it != tmp.rend(); ++it) {
			st.emplace_back(*it, u);
		}
	}
}

template <typename LabelT>
std::vector<size_t> InducedSubgraphView<LabelT>::sources() const
{
	std::vector<size_t> sources;
	const size_t n = _g->num_nodes();
	if (_node_list.empty()) return sources;

	// In-degree inside the view only.
	std::vector<size_t> indeg(n, 0);
	for (size_t u : _node_list) {
		// neighbors(u) already filters to nodes in the view
		for (const auto& e : neighbors(u)) {
			++indeg[e.to];
		}
	}

	// Start DFS from all sources in the view.
	for (size_t u : _node_list) {
		if (indeg[u] == 0) sources.push_back(u);
	}

	return sources;
}

template <typename LabelT>
void InducedSubgraphView<LabelT>::dfs_visit_from_sources(
	const std::function<void(size_t, std::optional<size_t>)>& cb) const
{
	const size_t n = _g->num_nodes();
	if (_node_list.empty()) return;

	// In-degree inside the view only.
	std::vector<size_t> indeg(n, 0);
	for (size_t u : _node_list) {
		// neighbors(u) already filters to nodes in the view
		for (const auto& e : neighbors(u)) {
			++indeg[e.to];
		}
	}

	std::vector<char> vis(n, 0);

	auto do_dfs = [&](size_t root) {
		using Frame = std::pair<size_t, std::optional<size_t>>;
		std::vector<Frame> st;
		st.emplace_back(root, std::nullopt);
		while (!st.empty()) {
			auto [u, parent] = st.back(); st.pop_back();
			if (u >= n || vis[u] || !contains_node(u)) continue;
			vis[u] = 1;

			cb(u, parent);

			std::vector<size_t> tmp;
			for (const auto& e : neighbors(u)) {
				const size_t v = e.to; // in view
				if (v < n && !vis[v] && contains_node(v)) tmp.push_back(v);
			}
			for (auto it = tmp.rbegin(); it != tmp.rend(); ++it) {
				st.emplace_back(*it, u);
			}
		}
	};

	// Start DFS from all sources in the view.
	for (size_t u : _node_list) {
		if (indeg[u] == 0 && !vis[u]) do_dfs(u);
	}

	// Optional fallback to cover any remaining nodes (e.g., cycles):
	// for (size_t u : _node_list) if (!vis[u]) do_dfs(u);
}

// ---- explicit instantiations ----------------------------------------------


template class Graph<double>;
template class Graph<py::object>;

template class InducedSubgraphView<double>;
template class InducedSubgraphView<py::object>;


/*
 * PYBIND11 MODULE
 */

void init_submodule_graphs(py::module_& m) {
        py::module_ graphs = m.def_submodule("graphs", "Graphs module.");

	using Label = py::object;
	using GraphPy = Graph<Label>;

	py::class_<GraphPy::Edge>(graphs, "Edge")
		.def_readonly("to", &GraphPy::Edge::to)
		.def_readonly("label", &GraphPy::Edge::label);

	py::class_<GraphPy>(graphs, "Graph")
		.def(py::init<bool>(), py::arg("directed")=true)
		.def_static("with_nodes", &GraphPy::WithNodes, py::arg("n"), py::arg("directed")=true)
		.def("add_node", &GraphPy::add_node)
		.def("add_nodes", &GraphPy::add_nodes)
		.def("remove_node", &GraphPy::remove_node)
		.def("add_edge", &GraphPy::add_edge)
		.def("remove_edge", &GraphPy::remove_edge)
		.def_property_readonly("num_nodes", &GraphPy::num_nodes)
		.def_property_readonly("directed", &GraphPy::directed)
		.def("alive", &GraphPy::alive)
		.def("neighbors", (const std::vector<GraphPy::Edge>& (GraphPy::*)(size_t) const)&GraphPy::neighbors,
		     py::arg("u"), py::return_value_policy::reference_internal)
		// .def("bfs", [](const GraphPy &g, size_t s){ auto r = g.bfs(s); return py::dict("parent"_a=r.parent, "dist"_a=r.dist); })
		// .def("dfs", &GraphPy::dfs)
		// .def("dijkstra", [](const GraphPy &g, size_t s, py::function weight_fn){
		// 	auto result = g.dijkstra(s, [weight_fn](const Label &lbl){ return weight_fn(lbl).cast<double>(); });
		// 	return py::dict("dist"_a=result.dist, "parent"_a=result.parent);
		// })
		.def("topological_sort", &GraphPy::topological_sort);
}
