#ifndef IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
#define IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "perceptron-layer.hpp"
#include "parametric-rectified-linear-unit.hpp"

#include <Eigen/Eigen>
#include <string>
#include <vector>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////////// | Class: MultiLayerPerceptron <
    class MultiLayerPerceptron final
    {
    public:
        //====================================================== | Structures <<
        struct TrainingResults;

        struct TestingResults;
        struct TestingResultsPerExample;

        //======================================================= | Behaviour <<
        //--------------------------------------------------------- | Static <<<
        static void initialiseRandomNumberGenerator
                (int const &seed);

        //--------------------------------------------------- | Constructors <<<
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

        TrainingResults train
                (std::vector<TrainingExample> const &trainingExamples,
                 std::vector<TrainingExample> const &testingExamples,
                 int numberOfEpochs,
                 double costGoal,
                 double learningCoefficient,
                 double learningCoefficientChange = 0.0,
                 double momentumCoefficient = 0.0,
                 bool shuffleTrainingData = true,
                 int epochInterval = 1);

        TestingResults test // TODO: Rename Training to Testing
                (std::vector<TrainingExample> const &testingExamples) const;

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
        std::vector<Eigen::VectorXd> feedForwardPerLayer
                (Eigen::VectorXd const &inputs) const;

        std::vector<Eigen::VectorXd> backpropagateErrorsPerLayer
                (std::vector<Eigen::VectorXd> const &inputsPerLayer,
                 Eigen::VectorXd const &errors) const;

    };

    //============================ | Class: MultiLayerPerceptron | Structures <<
    //----------------------------------------- | Structure: TrainingResults <<<
    struct MultiLayerPerceptron::TrainingResults
    {
        int epochInterval;
        std::vector<double> costPerEpochInterval;
        std::vector<double> accuracyTraining;
        std::vector<double> accuracyTesting;
    };

    //------------------------------------------ | Structure: TestingResults <<<
    struct MultiLayerPerceptron::TestingResults
    {
        double globalCost;
        std::vector<TestingResultsPerExample> testingResultsPerExample;
    };

    //-------------------------------- | Structure: TestingResultsPerExample <<<
    struct MultiLayerPerceptron::TestingResultsPerExample
    {
        std::vector<Eigen::VectorXd> neurons;
        Eigen::VectorXd targets;
        std::vector<Eigen::VectorXd> errors;
        double cost;
    };

}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_MULTI_LAYER_PERCEPTRON_HPP
