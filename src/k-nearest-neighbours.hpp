#ifndef IAD_2A_K_NEAREST_NEIGHBOURS_HPP
#define IAD_2A_K_NEAREST_NEIGHBOURS_HPP

#include "training-example.hpp"
#include <vector>
#include <Eigen/Eigen>

namespace NeuralNetworks
{
    class KNearestNeighbours
    {
    public:
        struct TestingResults;
        struct TestingResultsPerExample;

        KNearestNeighbours
                (int const k,
                 std::vector<TrainingExample> const &examples);

        Eigen::VectorXd operator()
                (Eigen::VectorXd const &inputs) const;

        TestingResults test
                (std::vector<TrainingExample> const &testingExamples) const;



    private:
        int const k;
        std::vector<TrainingExample> const examples;
    };

        //------------------------------------------ | Structure: TestingResults <<<
        struct KNearestNeighbours::TestingResults
        {
            double globalCost;
            std::vector<TestingResultsPerExample> testingResultsPerExample;
        };

        //-------------------------------- | Structure: TestingResultsPerExample <<<
        struct KNearestNeighbours::TestingResultsPerExample
        {
            std::vector<Eigen::VectorXd> neurons;
            Eigen::VectorXd targets;
            Eigen::VectorXd errors;
            double cost;
        };

    }

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_K_NEAREST_NEIGHBOURS_HPP
