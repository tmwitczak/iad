//
// Created by Tomasz Witczak on 31.05.2019.
//

#ifndef IAD_2A_NEURAL_NETWORK_LAYER_HPP
#define IAD_2A_NEURAL_NETWORK_LAYER_HPP

#include <Eigen/Eigen>
#include "cloneable.hpp"

namespace NeuralNetworks
{
    class NeuralNetworkLayer
            : private Cloneable<NeuralNetworkLayer>
    {
    public:
        //======================================================= | Behaviour <<
        //----------------------------------------------------- | Destructor <<<
        ~NeuralNetworkLayer
                () noexcept override = 0;

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<NeuralNetworkLayer> clone
                () const override = 0;

        operator std::unique_ptr<NeuralNetworkLayer>
                () const override
        {
            return std::move(clone());
        }

        //------------------------------------------------------ | Operators <<<
        virtual Eigen::VectorXd operator()
                (Eigen::VectorXd const &inputs) const = 0;

        //------------------------------------------------- | Main behaviour <<<
        virtual Eigen::VectorXd calculateOutputs
                (Eigen::VectorXd const &inputs) const = 0;

        virtual Eigen::VectorXd activate
                (Eigen::VectorXd const &outputs) const = 0;

        virtual Eigen::VectorXd calculateOutputsDerivative
                (Eigen::VectorXd const &outputs) const = 0;

        virtual Eigen::VectorXd feedForward
                (Eigen::VectorXd const &inputs) const = 0;

        virtual Eigen::VectorXd backpropagate
                (Eigen::VectorXd const &inputs,
                 Eigen::VectorXd const &errors,
                 Eigen::VectorXd const &outputs,
                 Eigen::VectorXd const &outputsDerivative) const = 0;

        virtual void calculateNextStep
                (Eigen::VectorXd const &inputs,
                 Eigen::VectorXd const &errors,
                 Eigen::VectorXd const &outputs,
                 Eigen::VectorXd const &outputsDerivative) = 0;

        virtual void update
                (double learningCoefficient,
                 double momentumCoefficient) = 0;

        virtual void saveToFile
                (std::string const &filename) const = 0;

        //--------------------------------------------------------- | Traits <<<
        virtual int numberOfInputs
                () const = 0;

        virtual int numberOfOutputs
                () const = 0;

    protected:
        //======================================================= | Behaviour <<
        //--------------------------------------------------- | Constructors <<<
        NeuralNetworkLayer
                () = default;

        NeuralNetworkLayer
                (NeuralNetworkLayer const &) = default;

        NeuralNetworkLayer
                (NeuralNetworkLayer &&) = default;

        //------------------------------------------------------ | Operators <<<
        NeuralNetworkLayer &operator=
                (NeuralNetworkLayer const &) = default;

        NeuralNetworkLayer &operator=
                (NeuralNetworkLayer &&) = default;
    };
}

#endif //IAD_2A_NEURAL_NETWORK_LAYER_HPP
