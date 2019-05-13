#ifndef IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
#define IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "perceptron-layer.hpp"
#include "parametric-rectified-linear-unit.hpp"

#include <Eigen>
#include <string>
#include <vector>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////////// | Class: MultiLayerPerceptron <
    class MultiLayerPerceptron final
    {
    public:
        //========================================================= | Methods <<
        //------------------------------------------------- | Static methods <<<
        static void initialiseRandomSeed
                (int const &seed);

        //--------------------------------------------------- | Constructors <<<
        explicit MultiLayerPerceptron
                (std::vector<int> const &numberOfNeuronsPerLayer,
                 std::vector<bool> const &enableBiasPerLayer = {});

        explicit MultiLayerPerceptron
                (std::string const &filename);

        //------------------------------------------------------ | Operators <<<
        Eigen::VectorXd operator()
                (Eigen::VectorXd const &inputs) const;

        //------------------------------------------------- | Main behaviour <<<
        Eigen::VectorXd feedForward
                (Eigen::VectorXd const &inputs) const;

        /*TrainingResults*/void train
                (std::vector<TrainingExample> const &trainingExamples,
                 int const numberOfEpochs,
                 double const costGoal,
                 double learningCoefficient,
                 double const learningCoefficientChange = 0.0,
                 double const momentumCoefficient = 0.0,
                 bool const shuffleTrainingData = true);

        void saveToFile
                (std::string const &filename) const;

        void readFromFile
                (std::string const &filename);

    private:
        //========================================================== | Fields <<
        std::vector<PerceptronLayer> layers;

        //========================================================= | Methods <<
        //------------------------------------------------- | Helper methods <<<
        std::vector<PerceptronLayer> createLayers
                (std::vector<int> const &numberOfNeuronsPerLayer,
                 std::vector<bool> enableBiasPerLayer = {}) const;

        std::vector<Eigen::VectorXd> feedForwardPerLayer
                (Eigen::VectorXd const &inputs) const;

        std::vector<Eigen::VectorXd> backpropagateErrorsPerLayer
                (std::vector<Eigen::VectorXd> const &inputsPerLayer,
                 Eigen::VectorXd const &errors) const;

    };

}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
