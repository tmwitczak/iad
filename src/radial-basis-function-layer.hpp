#ifndef IAD_2A_RADIAL_BASIS_FUNCTION_LAYER_HPP
#define IAD_2A_RADIAL_BASIS_FUNCTION_LAYER_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "activation-function.hpp"
#include "sigmoid.hpp"
#include "rectified-linear-unit.hpp"
#include "training-example.hpp"
#include "identity.hpp"
#include "neural-network-layer.hpp"

#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <cereal/access.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include "eigen-cereal.hpp"

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////// | Class: RadialBasisFunctionLayer <
    class RadialBasisFunctionLayer
            : public NeuralNetworkLayer
    {
    public:

        Eigen::VectorXd getBiases()
        {
            return biases;
        }
        void setWeights(Eigen::MatrixXd const &w)
        {
            weights = w;
        }
        Eigen::MatrixXd getWeights()
        {
            return weights;
        }
        //========================================================= | Methods <<
        //------------------------------------------------- | Static methods <<<
        static void initialiseRandomNumberGenerator
                (int seed);

        //--------------------------------------------------- | Constructors <<<
        RadialBasisFunctionLayer
                ();

        explicit RadialBasisFunctionLayer
                (int numberOfInputs,
                 int numberOfOutputs,
                 ActivationFunction const &activationFunction = Identity {});

        explicit RadialBasisFunctionLayer
                (std::string const &filename);

        RadialBasisFunctionLayer
                (RadialBasisFunctionLayer const &);

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<NeuralNetworkLayer> clone
                () const final;

        //------------------------------------------------------ | Operators <<<
        Eigen::VectorXd operator()
                (Eigen::VectorXd const &inputs) const override;

        //------------------------------------------------- | Main behaviour <<<
        Eigen::VectorXd calculateOutputs
                (Eigen::VectorXd const &inputs) const override;

        Eigen::VectorXd activate
                (Eigen::VectorXd const &outputs) const override;

        Eigen::VectorXd calculateOutputsDerivative
                (Eigen::VectorXd const &outputs) const override;

        Eigen::VectorXd feedForward
                (Eigen::VectorXd const &inputs) const override;

        Eigen::VectorXd backpropagate
                (Eigen::VectorXd const &inputs,
                 Eigen::VectorXd const &errors,
                 Eigen::VectorXd const &outputs,
                 Eigen::VectorXd const &outputsDerivative) const override;

        void calculateNextStep
                (Eigen::VectorXd const &inputs,
                 Eigen::VectorXd const &errors,
                 Eigen::VectorXd const &outputs,
                 Eigen::VectorXd const &outputsDerivative) override;

        void update
                (double learningCoefficient,
                 double momentumCoefficient) override;

        void saveToFile
                (std::string const &filename) const override;

        //--------------------------------------------------------- | Traits <<<
        int numberOfInputs
                () const override;

        int numberOfOutputs
                () const override;

    private:
        //============================================================ | Data <<
        Eigen::MatrixXd weights, deltaWeights, momentumWeights;
        Eigen::VectorXd biases, deltaBiases, momentumBiases;
        std::unique_ptr<ActivationFunction> activationFunction;
        int currentNumberOfSteps;

        //======================================================= | Behaviour <<
        //-------------------------------------------------- | Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const
        {
            archive(weights, deltaWeights, momentumWeights,
                    biases, deltaBiases, momentumBiases,
                    activationFunction,
                    currentNumberOfSteps);
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
            archive(weights, deltaWeights, momentumWeights,
                    biases, deltaBiases, momentumBiases,
                    activationFunction,
                    currentNumberOfSteps);
        }

        //----------------------------------------------- | Helper functions <<<
        void applyAverageOfDeltaStepsToMomentumStep
                (double learningCoefficient,
                 double momentumCoefficient);

        void applyMomentumStepToWeightsAndBiases
                ();

        void resetStepData
                ();


        double calculateDerivativeOfOutputWithRespectToInput(
                double const input,
                double const output,
                double const weight,
                double const bias) const;

        double calculateDerivativeOfCostWithRespectToOutput(double const
        error) const;
        double calculateDerivativeOfCostWithRespectToInput
                (double const input,
                 Vector const &weights,
                 Vector const &errors,
                 Vector const &outputs,
                 Vector const &outputsDerivative) const;

        double calculateDerivativeOfOutputWithRespectToWeight(
                double const input,
                double const output,
                double const weight,
                double const bias) const;

        double calculateDerivativeOfOutputWithRespectToBias(
                Vector const &inputs,
                double const output,
                Vector const &weights,
                double const bias) const;
    };
}

//////////////////////////////////////// | cereal: Polymorphic type registration
CEREAL_REGISTER_TYPE(NeuralNetworks::RadialBasisFunctionLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::NeuralNetworkLayer,
                                     NeuralNetworks::RadialBasisFunctionLayer)

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_RADIAL_BASIS_FUNCTION_LAYER_HPP
