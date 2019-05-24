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
            examples { examples }
    {
    }

    Vector KNearestNeighbours::operator()
            (Vector const &inputs) const
    {
        // Compute distances
        std::vector
                <std::pair
                        <double, std::reference_wrapper<TrainingExample const>>>
                distances(examples.size(),
                          std::make_pair(0.0, std::cref(examples.at(0))));

        for (auto[example, distance] = std::make_tuple(examples.cbegin(),
                                                       distances.begin());
             example != examples.cend();
             ++example, ++distance)
        {
            distance->first = (example->inputs - inputs).norm();
            distance->second = std::cref(*example);
        }

        // Sort distances
        using iksde = std::pair<double,
                std::reference_wrapper<TrainingExample const>>;
        static auto const comparator
                = [](iksde const &a, iksde const &b) -> bool
                {
                    return a.first < b.first;
                };
        std::sort(distances.begin(),
                  distances.end(),
                  comparator);

        // Average the outputs
        Vector sumOfKTargets { distances.cbegin()->second.get().outputs };
        for (auto[i, distance]= std::make_tuple(0,
                                                distances.cbegin() + 1);
             i < k && i < distances.size();
             ++i, ++distance)
            sumOfKTargets.noalias() += distance->second.get().outputs;

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