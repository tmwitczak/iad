#ifndef IAD_2A_CLONEABLE_HPP
#define IAD_2A_CLONEABLE_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include <memory>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    /////////////////////////////////////////////////// | Interface: Cloneable <
    template <typename T>
    class Cloneable
    {
    public:
        //========================================================= | Methods <<
        //------------------------------------------------- | Main behaviour <<<
        virtual std::unique_ptr<T> clone
                () const = 0;

    protected:
        //========================================================= | Methods <<
        //--------------------------------------------------- | Constructors <<<
        Cloneable
                () = default;

        Cloneable
                (Cloneable const &) = default;

        Cloneable
                (Cloneable &&) noexcept = default;

        //------------------------------------------------------ | Operators <<<
        Cloneable &operator=
                (Cloneable const &) = default;

        Cloneable &operator=
                (Cloneable &&) noexcept = default;

        //----------------------------------------------------- | Destructor <<<
        virtual ~Cloneable
                () noexcept = 0;
    };

    template <typename T>
    inline Cloneable<T>::~Cloneable
            () noexcept = default;
}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_CLONEABLE_HPP
