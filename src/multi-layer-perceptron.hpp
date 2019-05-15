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
        //======================================================= | Behaviour <<
        //--------------------------------------------------------- | Static <<<
        static void initialiseRandomNumberGenerator
                (int const &seed);

        //--------------------------------------------------- | Constructors <<<
        MultiLayerPerceptron
                () = default;

        explicit MultiLayerPerceptron
                (std::vector<int> const &numberOfNeurons,
                 std::vector<bool> const &enableBiasPerLayer);

        explicit MultiLayerPerceptron
                (std::string const &filename);

        //------------------------------------------------------ | Operators <<<
        Eigen::VectorXd operator()
                (Eigen::VectorXd const &inputs) const;

        //----------------------------------------------------------- | Main <<<
        Eigen::VectorXd feedForward
                (Eigen::VectorXd const &inputs) const;

        /*TrainingResults*/void train
                (std::vector<TrainingExample> const &trainingExamples,
                 int numberOfEpochs,
                 double costGoal,
                 double learningCoefficient,
                 double learningCoefficientChange = 0.0,
                 double momentumCoefficient = 0.0,
                 bool shuffleTrainingData = true);

        void saveToFile
                (std::string const &filename) const;

        void readFromFile
                (std::string const &filename);

    private:
        //============================================================ | Data <<
        std::vector<PerceptronLayer> layers;

        //======================================================= | Behaviour <<
        //-------------------------------------------------- | Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const;

        template <typename Archive>
        void load
                (Archive &archive);

        //----------------------------------------------- | Helper functions <<<
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
