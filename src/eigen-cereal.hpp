//
// Created by Tomasz Witczak on 31.05.2019.
//

#ifndef IAD_2A_EIGEN_CEREAL_HPP
#define IAD_2A_EIGEN_CEREAL_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <Eigen/Eigen>

using Array = Eigen::ArrayXd;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

////////////////////////////////////////////// | cereal: Archive specialisations
namespace cereal
{
    template <typename Archive>
    void save
            (Archive &archive,
             Matrix const &matrix)
    {
        int matrixRows = matrix.rows();
        int matrixColumns = matrix.cols();

        archive(matrixRows);
        archive(matrixColumns);

        archive(binary_data(matrix.data(),
                            matrixRows * matrixColumns * sizeof(double)));
    }

    template <typename Archive>
    void load
            (Archive &archive,
             Matrix &matrix)
    {
        int matrixRows = matrix.rows();
        int matrixColumns = matrix.cols();

        archive(matrixRows);
        archive(matrixColumns);

        matrix.resize(matrixRows, matrixColumns);

        archive(binary_data(matrix.data(),
                            matrixRows * matrixColumns * sizeof(double)));
    }

    template <typename Archive>
    void save
            (Archive &archive,
             Vector const &vector)
    {
        save(archive, Matrix { vector });
    }

    template <typename Archive>
    void load
            (Archive &archive,
             Vector &vector)
    {
        Matrix vectorAsMatrix;
        load(archive, vectorAsMatrix);
        vector = vectorAsMatrix;
    }
}

#endif //IAD_2A_EIGEN_CEREAL_HPP
