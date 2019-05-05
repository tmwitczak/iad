//
// Created by Tomasz Witczak on 20.04.2019.
//

#ifndef IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
#define IAD_2A_MULTI_LAYER_PERCEPTRON_HPP


#include <vector>
#include "single-layer-perceptron.hpp"
#include <Eigen>

////////////////////////////////////////////////////////////////////////////////
class MultiLayerPerceptron
{
public:
    static void initialiseRandomSeed(int const &seed)
    {
        SingleLayerPerceptron::initialiseRandomSeed(seed);
    }

    MultiLayerPerceptron(std::vector<int> const &numberOfNeuronsPerLayer)
            :
            layers { [&]
                     {
                         std::vector<SingleLayerPerceptron> layers;

                         for (auto i = numberOfNeuronsPerLayer.begin();
                              i != numberOfNeuronsPerLayer.end() - 1; i++)
                             layers.emplace_back(*i, *(i + 1));

                         return layers;
                     }() }
    {
    }

    Eigen::VectorXd feedForward
            (Eigen::VectorXd const &inputs) const
    {
        Eigen::VectorXd neurons = inputs;

        for (auto const &layer : layers)
            neurons = layer.feedForward(neurons);

        return neurons;
    }

    Eigen::VectorXd operator()
            (Eigen::VectorXd const &inputs) const
    {
        return feedForward(inputs);
    }

    void train
            (std::vector<TrainingExample> const &trainingExamples,
             int const &numberOfEpochs,
             double const &errorGoal,
             double const &learningRate,
             double const &momentum)
    {
        for (int epoch = 0; epoch < numberOfEpochs; epoch++)
        {
//            std::vector<Eigen::MatrixXd> deltaWeights
//                    { [&]
//                      {
//                          std::vector<Eigen::MatrixXd> deltaWeights;
//
//                          for (auto const &layer : layers)
//                              deltaWeights.emplace_back(
//                                      Eigen::MatrixXd::Zero(
//                                              layer.numberOfOutputs(),
//                                              layer.numberOfInputs()));
//
//                          return deltaWeights;
//                      }() };
//            vector <VectorXd> deltaBiases { [&]
//                                            {
//                                                vector <VectorXd> deltaBiases;
//
//                                                for (auto const &layer : layers)
//                                                    deltaBiases.push_back(
//                                                            VectorXd::Zero(
//                                                                    layer.getNumberOfOutputs()));
//
//                                                return deltaBiases;
//                                            }() };

            /*vector<VectorXd> errorsSum {[&]
            {
                vector<VectorXd> errorsSum;

                for (auto const &layer : layers)
                    errorsSum.push_back(VectorXd::Zero(layer.numberOfOutputs()));

                return errorsSum;
            }()};*/

            for (auto const &trainingExample : trainingExamples)
            {
                // Feed-forward
                std::vector<Eigen::VectorXd> outputsPerLayer
                        { calculateOutputsPerLayer(trainingExample) };

                // Calculate error (backpropagation)
                std::vector<Eigen::VectorXd> errors
                        { trainingExample.outputs - outputsPerLayer.back() };

                int x = layers.size() - 1;
                for (auto layer = layers.end() - 1;
                     layer != layers.begin(); layer--)
                {

                    errors.push_back
                            (layer->backpropagate(errors.back(),
                                                  outputsPerLayer.at(x)));

                    x--;

                    //VectorXd target{ VectorXd::Zero(layer->numberOfInputs()) };

                    /*for (int i = 0; i < layer->numberOfOutputs(); i++)
                    {
                        for (int j = 0; j < layer->numberOfInputs(); j++)
                        {
                            target(j) +=
                                (outputs.back().array() * (VectorXd::Ones(outputs.back().size()).array() - outputs.back().array())).matrix()
                                targets.back()(i);
                        }
                    }

                    targets.push_back(target);*/
                }
                std::reverse(errors.begin(), errors.end());
                /*for (int i = 0; i < errors.size(); i++)
                {
                    cout << outputs.at(i + 1) + errors.at(i) << "\n\n";
                }
                system("pause");*/

                //errorsSum += errors;


                //system("pause");

                /*for (int i = 0; i < outputs.size(); i++)
                    for (int j = 0; j < inputs.size(); j++)
                        deltaWeights(i, j) +=
                            2 * learningRate * inputs(j) * errors(i);*/

                /*for (int i = 0; i < outputs.size(); i++) {
                    deltaBiases(i) +=
                        2 * learningRate * errors(i);
                }*/

                // Update weights and biases
                for (int i = layers.size() - 1; i >= 0; i--)
                    layers.at(i).train({{ outputsPerLayer.at(i),
                                                outputsPerLayer.at(i + 1)
                                                + errors.at(i) }}, 1, 0.0,
                                       learningRate, momentum);
            }



            //cout << deltaWeights.array() / trainingExamples.size() << "\n\n";

            /*weights += (deltaWeights / trainingExamples.size());
            biases += (deltaBiases / trainingExamples.size());

            double cost = ((errorsSum.array() / trainingExamples.size())
                * (errorsSum.array() / trainingExamples.size())).sum();

            if (cost < errorGoal) break;

            cout << setw(3) << epoch << " | " << cost << endl;*/
        }

        //



        //	cout << heh << " | " << 0.5 * (errors.array() * errors.array()).sum() << endl;
        //}

        //// Back-propagate
        //

        /*cout << "Input: \n" << inputs << "\n\n";
        cout << "Output: \n" << outputs << "\n\n";
        cout << "Targets: \n" << targets << "\n\n";
        cout << "Errors: \n" << errors << "\n\n";
        cout << "Weights: \n" << weights << "\n\n";
        cout << "Bias: \n" << biases << "\n\n";*/
    }

private:
    std::vector<SingleLayerPerceptron> layers;

    std::vector<Eigen::VectorXd> calculateOutputsPerLayer
            (TrainingExample const &trainingExample) const
    {
        std::vector<Eigen::VectorXd> outputs { trainingExample.inputs };

        for (auto const &layer : layers)
            outputs.emplace_back(layer.feedForward(outputs.back()));

        return outputs;
    }
};


#endif //IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
