#ifndef IAD_2A_AFFINE_LAYER_HPP
#define IAD_2A_AFFINE_LAYER_HPP
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
    ///////////////////////////////////////////////////// | Class: AffineLayer <
    class AffineLayer
            : public NeuralNetworkLayer
    {
    public:
        //========================================================= | Methods <<
        //------------------------------------------------- | Static methods <<<
        static void initialiseRandomNumberGenerator
                (int seed);

        //--------------------------------------------------- | Constructors <<<
        AffineLayer
                ();

        explicit AffineLayer
                (int numberOfInputs,
                 int numberOfOutputs,
                 ActivationFunction const &activationFunction,
                 bool enableBias);

        explicit AffineLayer
                (std::string const &filename);

        AffineLayer
                (AffineLayer const &);

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

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<NeuralNetworkLayer> clone
                () const override;

    private:
        //============================================================ | Data <<
        Eigen::MatrixXd weights, deltaWeights, momentumWeights;
        Eigen::VectorXd biases, deltaBiases, momentumBiases;
        std::unique_ptr<ActivationFunction> activationFunction;
        int currentNumberOfSteps;
        bool isBiasEnabled;

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
                    currentNumberOfSteps,
                    isBiasEnabled);
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
            archive(weights, deltaWeights, momentumWeights,
                    biases, deltaBiases, momentumBiases,
                    activationFunction,
                    currentNumberOfSteps,
                    isBiasEnabled);
        }

        //----------------------------------------------- | Helper functions <<<
        void applyAverageOfDeltaStepsToMomentumStep
                (double learningCoefficient,
                 double momentumCoefficient);

        void applyMomentumStepToWeightsAndBiases
                ();

        void resetStepData
                ();
    };

    ///////////////////////////////////////////// | Class: AffineLayerWithBias <
    class AffineLayerWithBias
            : public AffineLayer
    {
    public:
        //========================================================= | Methods <<
        //--------------------------------------------------- | Constructors <<<
        AffineLayerWithBias
                ();

        explicit AffineLayerWithBias
                (int numberOfInputs,
                 int numberOfOutputs,
                 ActivationFunction const &activationFunction = Identity {});

        explicit AffineLayerWithBias
                (std::string const &filename);

        AffineLayerWithBias
                (AffineLayerWithBias const &);

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<NeuralNetworkLayer> clone
                () const final;

    private:
        //======================================================= | Behaviour <<
        //-------------------------------------------------- | Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const
        {
            archive(*this);
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
            archive(*this);
        }
    };

    ///////////////////////////////////////////// | Class: AffineLayerWithBias <
    class AffineLayerWithoutBias
            : public AffineLayer
    {
    public:
        //========================================================= | Methods <<
        //--------------------------------------------------- | Constructors <<<
        AffineLayerWithoutBias
                ();

        explicit AffineLayerWithoutBias
                (int numberOfInputs,
                 int numberOfOutputs,
                 ActivationFunction const &activationFunction = Identity {});

        explicit AffineLayerWithoutBias
                (std::string const &filename);

        AffineLayerWithoutBias
                (AffineLayerWithoutBias const &);

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<NeuralNetworkLayer> clone
                () const final;

    private:
        //======================================================= | Behaviour <<
        //-------------------------------------------------- | Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const
        {
            archive(*this);
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
            archive(*this);
        }
    };
}

//////////////////////////////////////// | cereal: Polymorphic type registration
CEREAL_REGISTER_TYPE(NeuralNetworks::AffineLayer)
CEREAL_REGISTER_TYPE(NeuralNetworks::AffineLayerWithBias)
CEREAL_REGISTER_TYPE(NeuralNetworks::AffineLayerWithoutBias)

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::NeuralNetworkLayer,
                                     NeuralNetworks::AffineLayer)

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::AffineLayer,
                                     NeuralNetworks::AffineLayerWithBias)
//CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::NeuralNetworkLayer,
//                                     NeuralNetworks::AffineLayerWithBias)

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::AffineLayer,
                                     NeuralNetworks::AffineLayerWithoutBias)
//CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::NeuralNetworkLayer,
//                                     NeuralNetworks::AffineLayerWithoutBias)

////////////////////////////////////////////////////////////////////////////////
#endif //IAD_2A_AFFINE_LAYER_HPP
