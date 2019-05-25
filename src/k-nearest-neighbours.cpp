//
// Created by Tomasz Witczak on 24.05.2019.
//

#include "k-nearest-neighbours.hpp"
#include <map>
#include <functional>
#include <iostream>

using Vector = Eigen::VectorXd;

namespace NeuralNetworks
{
    KNearestNeighbours::KNearestNeighbours
            (int const k,
             std::vector<TrainingExample> const &examples)
            :
            k { k },
            examples { examples },
            distances { examples.size() }
    {
    }

    using iksde = std::pair<double, TrainingExample const *>;

    Vector KNearestNeighbours::operator()
            (Vector const &inputs) const
    {
        static auto const comparator
                = [](iksde const &a, iksde const &b) -> bool
                {
                    return a.first < b.first;
                };

        // Compute distances
        for (auto[example, distance, examplesCEnd]
             = std::make_tuple(examples.cbegin(),
                               distances.begin(),
                               examples.cend());
             example != examplesCEnd;
             ++example, ++distance)
        {
            distance->first = (example->inputs - inputs).norm();
            distance->second = &(*example);
        }

        // Sort distances
        std::sort(distances.begin(),
                  distances.end(),
                  comparator);

        // Average the outputs
        Vector sumOfKTargets { distances.cbegin()->second->outputs };
        for (auto[i, distance]= std::make_tuple(k,
                                                distances.cbegin() + 1);
             i--;
             ++distance)
        {
            sumOfKTargets.noalias() += distance->second->outputs;
        }

        return sumOfKTargets / k;
    }

    KNearestNeighbours::TestingResults KNearestNeighbours::test
            (std::vector<TrainingExample> const &testingExamples) const
    {
        // Prepare results
        TestingResults testingResults;
        testingResults.globalCost = 0.0;

        int currentExampleNumber = 1;

        // Test the network
        for (auto const &testingExample
                : testingExamples)
        {
            std::cout << "example: " << currentExampleNumber++ << "\r";

            auto const &inputs
                    = testingExample.inputs;

            Vector const outputs
                    = this->operator()(inputs);

            std::vector<Vector> neurons
                    { inputs, outputs };

            auto const &targets
                    = testingExample.outputs;

            auto const errors
                    = targets - outputs;

            // Calculate cost
            double cost = errors.array().square().sum();
            testingResults.globalCost += cost;

            // Save testing results per example
            testingResults.testingResultsPerExample.push_back
                    ({ neurons,
                       targets,
                       errors,
                       cost });
        }

        // Average out global error
        testingResults.globalCost /= testingExamples.size();

        // Return testing report
        return testingResults;
    }
}