//----------------------------------------------------------------------------------------------------------------------
//! @file       cc/any.h
//! @brief      Implements the dynamic type `any`.  Adapted from Christopher Diggins' work.
//! @author     Matt Davies, Christopher Diggins, Pablo Aguilar, Kevlin Henney
//! @copyright  Copyright (C)2018 Bit-7 Technology\n
//!             Copyright (C)2005-2011 Christopher Diggins\n
//!             Copyright (C)2005 Pablo Aguilar\n
//!             Copyright (C)2001 Kevlin Henney
//----------------------------------------------------------------------------------------------------------------------

#pragma once

#include <cc/types.h>
#include <utility>

namespace cc
{

    class any;

    namespace impl
    {

        //--------------------------------------------------------------------------------------------------------------
        //! An exception that is thrown when a cast fails.
        
        struct BadAnyCast {};

        //--------------------------------------------------------------------------------------------------------------
        //! A special type that signifies an empty any value.
        
        struct EmptyAny {};

        //--------------------------------------------------------------------------------------------------------------
        //! A policy base-class that implements certain operations required for any implementation.
        
        struct AnyPolicyBase
        {
            virtual void staticDelete(void** x) = 0;
            virtual void copyFromValue(void const* src, void** dst) = 0;
            virtual void clone(void* const* src, void** dst) = 0;
            virtual void move(void* const* src, void** dst) = 0;
            virtual void* getValue(void** src) = 0;
            virtual i64 getSize() = 0;
        };

        //--------------------------------------------------------------------------------------------------------------
        //! Typed policy specialisation base class.
        
        template <typename T>
        struct TypedAnyPolicy : AnyPolicyBase
        {
            i64 getSize() override { return sizeof(T); }
        };

        //--------------------------------------------------------------------------------------------------------------
        //! Typed policy specialisation for small types.
        
        template <typename T>
        struct SmallTypedAnyPolicy : TypedAnyPolicy<T>
        {
            void staticDelete(void** x) override { }
            void copyFromValue(void const* src, void** dst) override { new(dst) T(*reinterpret_cast<T const *>(src)); }
            void clone(void* const* src, void** dst) override { *dst = *src; }
            void move(void* const* src, void** dst) override { *dst = *src; }
            void* getValue(void** src) override { return reinterpret_cast<void *>(src); }
        };

        //--------------------------------------------------------------------------------------------------------------
        //! Typed policy specialisation for large types.
        
        template <typename T>
        struct BigTypedAnyPolicy : TypedAnyPolicy<T>
        {
            void staticDelete(void** x) override { if (*x) delete(*reinterpret_cast<T**>(x)); *x = nullptr; }
            void copyFromValue(void const* src, void** dst) override { *dst = new T(*reinterpret_cast<T const *>(src)); }
            void clone(void* const* src, void** dst) override { *dst = new T(*reinterpret_cast<T const *>(src)); }
            void move(void* const* src, void** dst) override
            {
                (*reinterpret_cast<T**>(dst))->~T();
                **reinterpret_cast<T**>(dst) = **reinterpret_cast<T* const*>(src);
            }
            void* getValue(void** src) override { return *src; }
        };

        //--------------------------------------------------------------------------------------------------------------
        //! A structure to choose a policy based on type.
        template <typename T>
        struct ChoosePolicy
        {
            // Cannot be of type any.
            static_assert(!std::is_same<T, any>::value);
            using type = BigTypedAnyPolicy<T>;
        };

        //--------------------------------------------------------------------------------------------------------------
        //! A structure to choose a policy based on pointer types.
        
        template <typename T>
        struct ChoosePolicy<T*>
        {
            using type = SmallTypedAnyPolicy<T*>;
        };

        //--------------------------------------------------------------------------------------------------------------
        //! A macro to specialise small types.
        
        #define SMALL_POLICY(TYPE) template<> struct ChoosePolicy<TYPE> { using type = SmallTypedAnyPolicy<TYPE>; }

        SMALL_POLICY(i8);
        SMALL_POLICY(i16);
        SMALL_POLICY(i32);
        SMALL_POLICY(i64);
        SMALL_POLICY(u8);
        SMALL_POLICY(u16);
        SMALL_POLICY(u32);
        SMALL_POLICY(u64);
        SMALL_POLICY(f32);
        SMALL_POLICY(f64);
        SMALL_POLICY(bool);

        #undef SMALL_POLICY

        //--------------------------------------------------------------------------------------------------------------
        //! This function will return a different policy for each type.
        
        template <typename T>
        AnyPolicyBase* getAnyPolicy()
        {
            static typename ChoosePolicy<T>::type policy;
            return &policy;
        };
        
        //--------------------------------------------------------------------------------------------------------------

    }

    //------------------------------------------------------------------------------------------------------------------
    //! Represents a dynamic type.

    class any
    {
    public:
        //--------------------------------------------------------------------------------------------------------------
        //! Empty constructor

        any()
            : m_policy(impl::getAnyPolicy<impl::EmptyAny>())
            , m_object(nullptr)
        {}

        //--------------------------------------------------------------------------------------------------------------
        //! Initialising constructor

        template <typename T>
        any(const T& x) : any()
        {
            assign(x);
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Copy constructor

        any(const any& x) : any()
        {
            assign(x);
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Destructor

        ~any()
        {
            m_policy->staticDelete(&m_object);
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Assignment from another any.

        any& assign(const any& x)
        {
            clear();
            m_policy = x.m_policy;
            m_policy->clone(&x.m_object, &m_object);
            return *this;
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Assignment from another value.

        template <typename T>
        any& assign(const T& x)
        {
            clear();
            m_policy = impl::getAnyPolicy<T>();
            m_policy->copyFromValue(&x, &m_object);
            return *this;
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Assignment operator.

        template <typename T>
        any& operator= (const T& x)
        {
            return assign(x);
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Assignment operator specialisation for string literals.

        any& operator= (const char * x)
        {
            return assign(x);
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Swap function.

        any& swap(any& x)
        {
            std::swap(m_policy, x.m_policy);
            std::swap(m_object, x.m_object);
            return *this;
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Cast operator.
        //
        //! You can only cast to the original type.

        template <typename T>
        T& cast()
        {
            if (m_policy != impl::getAnyPolicy<T>()) throw impl::BadAnyCast();
            T* r = reinterpret_cast<T*>(m_policy->getValue(&m_object));
            return *r;
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Returns true if the any has no value.

        bool empty() const
        {
            return m_policy == impl::getAnyPolicy<impl::EmptyAny>();
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Frees allocated memory and sets the value to an empty value.

        void clear()
        {
            m_policy->staticDelete(&m_object);
            m_policy = impl::getAnyPolicy<impl::EmptyAny>();
        }

        //--------------------------------------------------------------------------------------------------------------
        //! Returns true if the two types are the same.

        bool isSameType(const any& x) const
        {
            return m_policy == x.m_policy;
        }

        //--------------------------------------------------------------------------------------------------------------

    private:
        impl::AnyPolicyBase* m_policy;          //!< Reference to the policy used (also a unique type value).
        void* m_object;                         //!< The actual object that holds the value.
    };

} // namespace cc

