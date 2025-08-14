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

	std::vector<size_t> dfs(size_t s) const;

	struct SSSPResult { std::vector<double> dist; std::vector<std::optional<size_t>> parent; };
	SSSPResult dijkstra(size_t s, std::function<double(const LabelT&)> weight_fn) const;

	std::vector<size_t> topological_sort() const;

private:
	void check_node(size_t u) const;

	bool directed_;
	size_t nodes_alive_ = 0;
	std::vector<std::vector<Edge>> adj_;
	std::vector<bool> alive_;
};
