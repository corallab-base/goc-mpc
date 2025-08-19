#include "utils.hpp"


py::array_t<double> remainder_slice_1d(const py::array_t<double>& arr, unsigned int i) {
	// Extract shape info
	auto info = arr.request();
	int N = info.shape[0];

	if (i >= N)
		throw std::out_of_range("index out of bounds for input arr");

	// Length of the subarray
	int len = N - i;
	const double* start_ptr = static_cast<const double*>(info.ptr) + i;

	// Create a new view with adjusted pointer and shape (no copy)
	return py::array_t<double>(
		{len},                          // shape
		{sizeof(double)},               // stride
		start_ptr,                      // pointer to start
		arr                             // base object to keep alive
	);
}

py::array_t<double> remainder_slice_2d(const py::array_t<double>& arr, unsigned int i) {
	// Get 2D shape info
	auto info = arr.request();
	int N = info.shape[0];
	int D = info.shape[1];

	if (i >= N)
		throw std::out_of_range("index out of bounds for input arr");

	// Slice from row = phase to N (rows), all columns
	int num_rows = N - i;
	const double* start_ptr = static_cast<const double*>(info.ptr) + i * D;

	return py::array_t<double>(
		{num_rows, D},                          // shape: rows × columns
		{sizeof(double) * D, sizeof(double)},   // strides: row-major
		start_ptr,
		arr // keep original array alive
	);
}

/* reimann integral with a step size of 1. (a cumulative sum) */
py::array_t<double> integral(const py::array_t<double>& x) {
	const auto ndim = x.ndim();
	if (ndim == 1) {
		auto x_ = x.unchecked<1>();
		int N = x_.shape(0);
		py::array_t<double> y(N);
		auto y_ = y.mutable_unchecked<1>();

		double s = 0.0;
		for (int i = 0; i < N; ++i) {
			s += x_(i);
			y_(i) = s;
		}
		return y;
	}

	if (ndim == 2) {
		auto x_ = x.unchecked<2>();
		int rows = x_.shape(0);
		int cols = x_.shape(1);
		py::array_t<double> y({rows, cols});
		auto y_ = y.mutable_unchecked<2>();

		// Copy x into y
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				y_(i, j) = x_(i, j);

		// Row-wise cumulative sum
		for (int i = 0; i < rows; ++i)
			for (int j = 1; j < cols; ++j)
				y_(i, j) += y_(i, j - 1);

		// Column-wise cumulative sum
		for (int j = 0; j < cols; ++j)
			for (int i = 1; i < rows; ++i)
				y_(i, j) += y_(i - 1, j);

		return y;
	}

	if (ndim == 3) {
		auto x_ = x.unchecked<3>();
		int d0 = x_.shape(0), d1 = x_.shape(1), d2 = x_.shape(2);
		py::array_t<double> y({d0, d1, d2});
		auto y_ = y.mutable_unchecked<3>();

		// Copy x into y
		for (int i = 0; i < d0; ++i)
			for (int j = 0; j < d1; ++j)
				for (int k = 0; k < d2; ++k)
					y_(i, j, k) = x_(i, j, k);

		// Cumulative sums along each axis
		for (int i = 1; i < d0; ++i)
			for (int j = 0; j < d1; ++j)
				for (int k = 0; k < d2; ++k)
					y_(i, j, k) += y_(i - 1, j, k);

		for (int i = 0; i < d0; ++i)
			for (int j = 1; j < d1; ++j)
				for (int k = 0; k < d2; ++k)
					y_(i, j, k) += y_(i, j - 1, k);

		for (int i = 0; i < d0; ++i)
			for (int j = 0; j < d1; ++j)
				for (int k = 1; k < d2; ++k)
					y_(i, j, k) += y_(i, j, k - 1);

		return y;
	}

	throw std::runtime_error("integral: unsupported number of dimensions (only 1D–3D supported).");
}

py::array_t<double> prepend_1d(const py::array_t<double>& a, double v) {
	int N = a.shape(0);
	py::array_t<double> result({N + 1});
	auto result_ = result.mutable_unchecked<1>();
	auto a_ = a.unchecked<1>();

	result_(0) = v;
	for (int i = 0; i < N; ++i)
		result_(i + 1) = a_(i);

	return result;
}

py::array_t<double> prepend_row_2d(const py::array_t<double>& a, const py::array_t<double>& row) {
	if (a.ndim() != 2 || row.ndim() != 1)
		throw std::runtime_error("Expected 2D array and 1D row vector");

	int N = a.shape(0);
	int D = a.shape(1);

	if (row.shape(0) != D)
		throw std::runtime_error("Row shape mismatch with array columns");

	py::array_t<double> result({N + 1, D});
	auto result_ = result.mutable_unchecked<2>();
	auto a_ = a.unchecked<2>();
	auto row_ = row.unchecked<1>();

	// First row
	for (int j = 0; j < D; ++j)
		result_(0, j) = row_(j);

	// Remaining rows
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < D; ++j)
			result_(i + 1, j) = a_(i, j);

	return result;
}
