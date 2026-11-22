/**
 * pybind11 C++ interface to the canonical FastDMF simulator (Mediano, Luppi & Rosas / Herzog et al. 2024).
 * Compiles directly against upstream concog/fastdmf cpp/DMF.hpp + bundled Eigen without Boost.Python.
 */

#include "DMF.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using FArray = py::array_t<double, py::array::f_style | py::array::forcecast>;

static bool is_scalar_array(const FArray& arr) {
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        if (arr.shape(i) > 1) {
            return false;
        }
    }
    return true;
}

static py::tuple run_dmf(
    const py::dict& py_params,
    size_t nb_steps,
    size_t N,
    bool return_rate,
    bool return_bold
) {
    // Keep Fortran-contiguous double arrays alive for the entire simulation.
    std::vector<FArray> owned_arrays;
    owned_arrays.reserve(py_params.size());

    ParamStruct params;
    for (auto item : py_params) {
        std::string key = py::cast<std::string>(item.first);
        FArray arr = FArray::ensure(item.second);
        if (!arr) {
            throw std::invalid_argument("Parameter '" + key + "' cannot be cast to float64 ndarray");
        }
        owned_arrays.push_back(arr);
        const double* ptr = owned_arrays.back().data();
        params[key] = ptr;
        std::string isvec_key = "isvector_" + key;
        params[isvec_key] = is_scalar_array(owned_arrays.back()) ? nullptr : ptr;
    }

    checkParams(params);

    const double dtt = params["dtt"][0];
    const double tr = params["TR"][0];
    const size_t batch_size = static_cast<size_t>(params["batch_size"][0]);
    const size_t nb_bold_steps = static_cast<size_t>(nb_steps * dtt / tr);
    const size_t rate_size = return_rate ? nb_steps : (2 * batch_size);

    // Allocate column-major output buffers matching Eigen::MatrixXd layout.
    std::vector<py::ssize_t> rate_shape = {static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(rate_size)};
    std::vector<py::ssize_t> bold_shape = {
        static_cast<py::ssize_t>(N),
        static_cast<py::ssize_t>(return_bold ? nb_bold_steps : 1)
    };
    FArray rate_arr(rate_shape);
    FArray bold_arr(bold_shape);
    std::fill(rate_arr.mutable_data(), rate_arr.mutable_data() + rate_arr.size(), 0.0);
    std::fill(bold_arr.mutable_data(), bold_arr.mutable_data() + bold_arr.size(), 0.0);

    double* rate_ptr = rate_arr.mutable_data();
    double* bold_ptr = bold_arr.mutable_data();

    {
        py::gil_scoped_release release;
        DMFSimulator sim(params, nb_steps, N, return_rate, return_bold);
        sim.run(rate_ptr, bold_ptr);
    }

    return py::make_tuple(rate_arr, bold_arr);
}

PYBIND11_MODULE(_fastdmf_ext, m) {
    m.doc() = "pybind11 bindings to canonical C++ FastDMF simulator";
    m.def("run_dmf", &run_dmf,
          py::arg("params"),
          py::arg("nb_steps"),
          py::arg("N"),
          py::arg("return_rate") = false,
          py::arg("return_bold") = true);
}
