#include "graphs.hpp"

namespace py = pybind11;


template <typename LabelT>
Graph<LabelT>::Graph(bool directed) : directed_(directed) {}

template <typename LabelT>
Graph<LabelT> Graph<LabelT>::WithNodes(size_t n, bool directed) {
	Graph g(directed);
	g.add_nodes(n);
	return g;
}

template <typename LabelT>
size_t Graph<LabelT>::add_node() {
	size_t id = nodes_alive_;
	++nodes_alive_;
	adj_.emplace_back();
	alive_.push_back(true);
	return id;
}

template <typename LabelT>
void Graph<LabelT>::add_nodes(size_t n) {
	for (size_t i = 0; i < n; ++i) add_node();
}

template <typename LabelT>
void Graph<LabelT>::remove_node(size_t u) {
	check_node(u);
	if (!alive_[u]) return;
	alive_[u] = false;
	adj_[u].clear();
	if (!directed_) {
		for (auto &nbrs : adj_) {
			nbrs.erase(std::remove_if(nbrs.begin(), nbrs.end(), [&](const Edge &e){return e.to==u;}), nbrs.end());
		}
	}
}

template <typename LabelT>
void Graph<LabelT>::add_edge(size_t u, size_t v, const LabelT &label) {
	check_node(u); check_node(v);
	if (!alive_[u] || !alive_[v]) throw std::invalid_argument("add_edge: node is removed");
	adj_[u].push_back(Edge{v, label});
	if (!directed_ && u != v) {
		adj_[v].push_back(Edge{u, label});
	}
}

template <typename LabelT>
bool Graph<LabelT>::remove_edge(size_t u, size_t v) {
	check_node(u); check_node(v);
	auto &nbrs = adj_[u];
	auto it = std::find_if(nbrs.begin(), nbrs.end(), [&](const Edge &e){ return e.to==v; });
	bool removed = false;
	if (it != nbrs.end()) { nbrs.erase(it); removed = true; }
	if (!directed_) {
		auto &nbrs2 = adj_[v];
		auto it2 = std::find_if(nbrs2.begin(), nbrs2.end(), [&](const Edge &e){ return e.to==u; });
		if (it2 != nbrs2.end()) { nbrs2.erase(it2); removed = true; }
	}
	return removed;
}

template <typename LabelT>
size_t Graph<LabelT>::num_nodes() const { return adj_.size(); }

template <typename LabelT>
bool Graph<LabelT>::directed() const { return directed_; }

template <typename LabelT>
bool Graph<LabelT>::alive(size_t u) const { check_node(u); return alive_[u]; }

template <typename LabelT>
const std::vector<typename Graph<LabelT>::Edge>& Graph<LabelT>::neighbors(size_t u) const { check_node(u); return adj_[u]; }

template <typename LabelT>
std::vector<typename Graph<LabelT>::Edge>& Graph<LabelT>::neighbors(size_t u) { check_node(u); return adj_[u]; }

template <typename LabelT>
typename Graph<LabelT>::BFSResult Graph<LabelT>::bfs(size_t s) const {
	check_node(s);
	const int INF = std::numeric_limits<int>::max();
	std::vector<int> dist(num_nodes(), INF);
	std::vector<std::optional<size_t>> parent(num_nodes(), std::nullopt);
	if (!alive_[s]) return {std::move(parent), std::move(dist)};

	std::queue<size_t> q; q.push(s); dist[s]=0; parent[s]=s;
	while(!q.empty()){
		size_t u=q.front(); q.pop();
		for(const auto& e: adj_[u]){
			size_t v=e.to; if(!alive_[v]) continue;
			if(dist[v]==INF){ dist[v]=dist[u]+1; parent[v]=u; q.push(v);} }
	}
	return {std::move(parent), std::move(dist)};
}

template <typename LabelT>
std::vector<size_t> Graph<LabelT>::dfs(size_t s) const {
	check_node(s);
	std::vector<size_t> out;
	if (!alive_[s]) return out;
	std::vector<char> vis(num_nodes(), 0);
	std::stack<size_t> st; st.push(s);
	while(!st.empty()){
		size_t u=st.top(); st.pop();
		if(vis[u]||!alive_[u]) continue; vis[u]=1; out.push_back(u);
		const auto &nbrs = adj_[u];
		for(auto it = nbrs.rbegin(); it!=nbrs.rend(); ++it){ st.push(it->to);}
	}
	return out;
}

template <typename LabelT>
typename Graph<LabelT>::SSSPResult Graph<LabelT>::dijkstra(size_t s, std::function<double(const LabelT&)> weight_fn) const {
	check_node(s);
	const double INF = std::numeric_limits<double>::infinity();
	std::vector<double> dist(num_nodes(), INF);
	std::vector<std::optional<size_t>> parent(num_nodes(), std::nullopt);
	if (!alive_[s]) return {std::move(dist), std::move(parent)};

	using QItem = std::pair<double,size_t>;
	struct Cmp { bool operator()(const QItem& a, const QItem& b) const { return a.first > b.first; } };
	std::priority_queue<QItem, std::vector<QItem>, Cmp> pq;
	dist[s]=0.0; parent[s]=s; pq.emplace(0.0, s);

	while(!pq.empty()){
		auto [du,u]=pq.top(); pq.pop(); if(du!=dist[u]) continue; if(!alive_[u]) continue;
		for(const auto& e: adj_[u]){
			size_t v=e.to; if(!alive_[v]) continue;
			double w = weight_fn(e.label);
			if(w < 0) throw std::invalid_argument("dijkstra: negative edge weight");
			if(dist[v] > du + w){ dist[v] = du + w; parent[v] = u; pq.emplace(dist[v], v);} }
	}
	return {std::move(dist), std::move(parent)};
}

template <typename LabelT>
std::vector<size_t> Graph<LabelT>::topological_sort() const {
	if (!directed_) throw std::logic_error("topological_sort: only for directed graphs");
	const size_t n = num_nodes();
	std::vector<size_t> indeg(n, 0);
	for(size_t u=0; u<n; ++u){ if(!alive_[u]) continue; for(const auto &e: adj_[u]) if(alive_[e.to]) ++indeg[e.to]; }
	std::queue<size_t> q; for(size_t u=0; u<n; ++u) if(alive_[u] && indeg[u]==0) q.push(u);
	std::vector<size_t> order; order.reserve(n);
	while(!q.empty()){
		size_t u=q.front(); q.pop(); order.push_back(u);
		for(const auto &e: adj_[u]){ if(!alive_[e.to]) continue; if(--indeg[e.to]==0) q.push(e.to); }
	}
	size_t alive_count = 0; for(bool a: alive_) if(a) ++alive_count;
	if(order.size() != alive_count)
		throw std::runtime_error("Graph has at least one cycle among alive nodes");
	return order;
}

template <typename LabelT>
void Graph<LabelT>::check_node(size_t u) const {
	if (u >= adj_.size()) throw std::out_of_range("node id out of range");
}

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
