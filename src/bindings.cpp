#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "brstboost.h"

namespace py = pybind11;
using namespace brst;

PYBIND11_MODULE(_brstboost, m) {
    m.doc() = "BRSTBoost: Bayesian Residual State Theory gradient boosting";

    py::class_<BRSTBoost>(m, "BRSTBoost")
        .def(py::init<>())
        .def("fit",
            [](BRSTBoost& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,   py::array::c_style | py::array::forcecast> y,
               int n_estimators, double learning_rate, int max_depth,
               double reg_lambda, double subsample, int n_bins,
               double min_child_weight, std::vector<int> cat_features,
               int random_state)
            {
                auto Xb = X.request(); auto yb = y.request();
                int n = Xb.shape[0], D = Xb.shape[1];
                Params p;
                p.n_estimators = n_estimators;
                p.learning_rate = learning_rate;
                p.max_depth = max_depth;
                p.reg_lambda = reg_lambda;
                p.subsample = subsample;
                p.n_bins = n_bins;
                p.min_child_weight = min_child_weight;
                p.cat_features = cat_features;
                p.random_state = random_state;
                self.fit((float*)Xb.ptr, n, D, (int*)yb.ptr, p);
            },
            py::arg("X"), py::arg("y"),
            py::arg("n_estimators") = 200,
            py::arg("learning_rate") = 0.1,
            py::arg("max_depth") = 4,
            py::arg("reg_lambda") = 1.0,
            py::arg("subsample") = 0.8,
            py::arg("n_bins") = 32,
            py::arg("min_child_weight") = 1.0,
            py::arg("cat_features") = std::vector<int>{},
            py::arg("random_state") = 0)
        .def("predict_proba",
            [](const BRSTBoost& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> X)
            -> py::array_t<double>
            {
                auto Xb = X.request();
                int n = Xb.shape[0], D = Xb.shape[1];
                py::array_t<double> out({n});
                self.predict_proba((float*)Xb.ptr, n, D,
                                   (double*)out.request().ptr);
                return out;
            })
        .def("avg_k", &BRSTBoost::avg_k);
}
