#ifndef IAD_2A_NEURAL_NETWORK_HPP
#define IAD_2A_NEURAL_NETWORK_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "affine-layer.hpp"
#include "parametric-rectified-linear-unit.hpp"

#include <Eigen/Eigen>
#include <string>
#include <vector>

#include "neural-network-layer.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    /////////////////////////////////////////////////// | Class: NeuralNetwork <
    class NeuralNetwork final
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
//        explicit NeuralNetwork
//                (std::vector<int> const &numberOfNeurons,
//                 std::vector<bool> const &enableBiasPerLayer);

        explicit NeuralNetwork
                (std::vector<std::unique_ptr<NeuralNetworkLayer>> layers);

        explicit NeuralNetwork
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
        std::vector<std::unique_ptr<NeuralNetworkLayer>> layers;

        //======================================================= | Behaviour <<
        //-------------------------------------------------- | Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const
        {
            archive(layers);
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
            archive(layers);
        }

        //----------------------------------------------- | Helper functions <<<
//        std::vector<Eigen::VectorXd> feedForwardPerLayer
//                (Eigen::VectorXd const &inputs) const;
//
//        std::vector<Eigen::VectorXd> backpropagateErrorsPerLayer
//                (std::vector<Eigen::VectorXd> const &inputsPerLayer,
//                 Eigen::VectorXd const &errors) const;

    };

    //============================ | Class: NeuralNetwork | Structures <<
    //----------------------------------------- | Structure: TrainingResults <<<
    struct NeuralNetwork::TrainingResults
    {
        int epochInterval;
        std::vector<double> costPerEpochInterval;
        std::vector<double> accuracyTraining;
        std::vector<double> accuracyTesting;
    };

    //------------------------------------------ | Structure: TestingResults <<<
    struct NeuralNetwork::TestingResults
    {
        double globalCost;
        std::vector<TestingResultsPerExample> testingResultsPerExample;
    };

    //-------------------------------- | Structure: TestingResultsPerExample <<<
    struct NeuralNetwork::TestingResultsPerExample
    {
        std::vector<Eigen::VectorXd> neurons;
        Eigen::VectorXd targets;
        std::vector<Eigen::VectorXd> errors;
        double cost;
    };

}

////////////////////////////////////////////////////////////////////////////////
#endif //IAD_2A_NEURAL_NETWORK_HPP
