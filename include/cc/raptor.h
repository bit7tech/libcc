//----------------------------------------------------------------------------------------------------------------------
//! @file       raptor.h
//! @brief      Multi-threaded programming library.
//! @copyright  Copyright (C)2018 Bit-7 Technology, all rights reserved.
//! @author     Matt Davies
//----------------------------------------------------------------------------------------------------------------------

#pragma once

// #todo: Tidy this file up.  Put implementations to the section at the end.

#include <malloc.h>
#include <asio.hpp>

#include <atomic>
#include <memory>
#include <map>
#include <optional>
#include <unordered_set>

namespace cc
{

    //------------------------------------------------------------------------------------------------------------------
    // Constants

    static const int kDefaultDataQueueSize = 1024;

    //------------------------------------------------------------------------------------------------------------------

    class Scheduler;

    //------------------------------------------------------------------------------------------------------------------
    // Function trait utilities

    template <typename T>
    struct functionTraits
        : public functionTraits<decltype(&T::operator())>
    {};

    //! Function traits for pointers to const member functions.
    template <class ClassT, typename ReturnT, typename... ArgsT>
    struct functionTraits<ReturnT(ClassT::*)(ArgsT...) const>
    {
        using type = std::function<ReturnT(ArgsT...)>;
    };

    //! Function traits for pointers to member functions.
    template <class ClassT, typename ReturnT, typename... ArgsT>
    struct functionTraits<ReturnT(ClassT::*)(ArgsT...)>
    {
        using type = std::function<ReturnT(ArgsT...)>;
    };

    //! Function traits for C functions.
    template <typename ReturnT, typename... ArgsT>
    struct functionTraits<ReturnT(*)(ArgsT...)>
    {
        using type = std::function<ReturnT(ArgsT...)>;
    };

    template <typename CallableT>
    struct CreateFunction
    {
        using type = typename functionTraits<typename std::decay<CallableT>::type>::type;
    };

    template <typename CallableT>
    struct GetReturnType
    {
        using type = typename GetReturnType<typename CreateFunction<CallableT>::type>::type;
    };

    template <typename ReturnT, typename... ArgsT>
    struct GetReturnType<std::function<ReturnT(ArgsT...)>>
    {
        using type = ReturnT;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Holds a result for a task that can be accessed from another thread

    template <typename T>
    class Result
    {
    public:
        //! Construct a result with starts in an unset state.
        explicit Result()
            : m_state(State::Unset)
        {}

        //! Retrieve the value (blocking until value has been resolved).
        const T& get() const
        {
            while (getState() != State::Set) YieldProcessor();
            return m_value;
        }

        //! Set the value.
        bool set(const T& value)
        {
            bool success = setState(State::Unset, State::Value);
            if (success)
            {
                m_value = value;
                success = setState(State::Value, State::Set);
                assert(success);
            }
            return success;
        }

        enum class State : int
        {
            Unset,
            Error,
            Value,
            Set
        };

        //! Return the state of the result.
        State getState() const
        {
            return m_state.load(std::memory_order_acquire);
        }

        //! Set the state.
        //
        //! Returns true if the state was changed.  It will fail is the current state is not the expected one.
        bool setState(State expected, State newState)
        {
            return m_state.compare_exchange_strong(expected, newState, std::memory_order_acq_rel);
        }

    private:
        std::atomic<State>      m_state;
        T                       m_value;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Result object for void types

    template <>
    class Result<void>
    {
    public:
        //! Construct a result with starts in an unset state.
        explicit Result()
            : m_state(State::Unset)
        {}

        //! Signify that the task completed.
        bool set()
        {
            bool success = setState(State::Unset, State::Value);
            if (success)
            {
                success = setState(State::Value, State::Set);
                assert(success);
            }
            return success;
        }

        enum class State : int
        {
            Unset,
            Value,
            Set
        };

        //! Return the state of the result.
        State getState() const
        {
            return m_state.load(std::memory_order_acquire);
        }

        //! Set the state.
        //
        //! Returns true if the state was changed.  It will fail is the current state is not the expected one.
        bool setState(State expected, State newState)
        {
            return m_state.compare_exchange_strong(expected, newState, std::memory_order_acq_rel);
        }

    private:
        std::atomic<State>      m_state;

    };

    //------------------------------------------------------------------------------------------------------------------
    //! Base class for all templated tasks.

    class TaskBase
    {
    public:
        virtual void execute() = 0;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Owns a function that executes a task and manages a Result instance and continuations.

    template <typename T>
    class TaskImpl : public TaskBase
    {
    public:
        //! Construct a task implementation from a function
        TaskImpl(std::function<T()> func)
            : m_func(func)
        {
        }

        //! Block this thread until the result is available.
        void wait()
        {
            while (m_result.getState() != Result<T>::State::Set) YieldProcessor();
        }

        //! Return true if the task has finished and a result is available.
        bool hasFinished()
        {
            return m_result.getState() == Result<T>::State::Set;
        }

        //! Wait for the result, if necessary, and return it.
        T get()
        {
            return m_result.get();
        }

        //! Execute the function
        void execute() override
        {
            m_result.set(m_func());

        }

    private:
        std::function<T()>  m_func;
        Result<T>           m_result;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Owns a function that executes a task and manages a Result instance and continuations.  This is the no-result
    //! version of TaskImpl.

    template <>
    class TaskImpl<void> : public TaskBase
    {
    public:
        //! Construct a task implementation from a function
        TaskImpl(std::function<void()> func)
            : m_func(func)
        {
        }

        //! Block this thread until the result is available.
        void wait()
        {
            while (m_result.getState() != Result<void>::State::Set) YieldProcessor();
        }

        //! Return true if the task has finished and a result is available.
        bool hasFinished()
        {
            return m_result.getState() == Result<void>::State::Set;
        }

        //! Execute the function
        void execute() override
        {
            m_func();
            m_result.set();

        }

    private:
        std::function<void()>  m_func;
        Result<void>           m_result;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Wraps a TaskImpl as a value

    template <typename T>
    class Task
    {
    public:
        using type = T;

        Task(std::shared_ptr<TaskImpl<T>> task)
            : m_task(task)
        {

        }

        void wait()
        {
            return m_task->wait();
        }

        bool hasFinished()
        {
            return m_task->hasFinished();
        }

        T get()
        {
            return m_task->get();
        }

    private:
        std::shared_ptr<TaskImpl<T>>    m_task;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Base class for all queues.

    class Queue
    {
    public:
        Queue(Scheduler& scheduler)
            : m_scheduler(scheduler)
        {}

        template <typename CallableT>
        Task<typename GetReturnType<typename std::decay<CallableT>::type>::type>
            async(CallableT&& func)
        {
            using ReturnT = typename GetReturnType<typename::std::decay<CallableT>::type>::type;
            std::function<ReturnT()> f(func);
            auto task = std::make_shared<TaskImpl<ReturnT>>(func);
            pushTask(std::static_pointer_cast<TaskBase>(task));
            return Task(task);
        }

    protected:
        virtual void pushTask(const std::shared_ptr<TaskBase>& task) = 0;

        Scheduler& getScheduler() { return m_scheduler; }
        const Scheduler& getScheduler() const { return m_scheduler; }

    private:
        Scheduler & m_scheduler;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Concurrent queue that can hold tasks that can be executed concurrently.

    class ConcurrentQueue : public Queue
    {
    public:
        ConcurrentQueue(Scheduler& scheduler) : Queue(scheduler) {}

    protected:
        void pushTask(const std::shared_ptr<TaskBase>& task) override;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Serial queue that can hold tasks that cannot be executed concurrently with other tasks on the queue.

    class SerialQueue : public Queue
    {
    public:
        SerialQueue(Scheduler& scheduler);

    protected:
        void pushTask(const std::shared_ptr<TaskBase>& task) override;

    private:
        asio::io_context::strand  m_strand;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! The various queue types allowed on the scheduler.

    enum class QueueType
    {
        Concurrent,
        Serial,
        Timer,
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Manages worker threads and feeds them from created queues.

    class Scheduler
    {
    public:
        //! Scheduler construction.
        Scheduler(int numThreads = std::thread::hardware_concurrency())
            : m_ioService()
            , m_work(m_ioService)
            , m_defaultQueue(*this)
        {
            m_threads.reserve(numThreads);
            for (int i = 0; i < numThreads; ++i)
            {
                m_threads.emplace_back(std::bind(&Scheduler::workerThread, this));
            }
        }

        //! Scheduler destructor.
        ~Scheduler()
        {
            m_ioService.stop();
            for (auto& thread : m_threads)
            {
                thread.join();
            }
        }

        //! Create a queue to execute tasks on.
        std::shared_ptr<Queue> createQueue(QueueType type)
        {
            switch (type)
            {
            case QueueType::Concurrent:
                return std::make_shared<ConcurrentQueue>(*this);

            case QueueType::Serial:
                return std::make_shared<SerialQueue>(*this);

            default:
                assert(0);
                return {};
            }
        }

        //! Return a reference to the default concurrent queue
        ConcurrentQueue& getDefaultQueue();

        //! Traits for when all functions.
        //
        //! The iterator type is container<Task<T>>::iterator.
        template <typename IterT>
        struct WhenAllTraits
        {
            using task_type = typename IterT::value_type;
            using value_type = typename task_type::type;
        };

        //! Returns a task that returns vector<T> where T is the result type by waiting for the results
        //! of all the tasks given by the iterators.
        template <typename IterT>
        Task<std::vector<typename WhenAllTraits<IterT>::value_type>> when_all(IterT begin, IterT end)
        {
            using T = WhenAllTraits<IterT>::value_type;
            return m_defaultQueue.async([begin, end]() -> std::vector<T>
            {
                std::vector<T> results;
                results.reserve(end - begin);
                for (std::vector<Task<T>>::iterator it = begin; it != end; ++it)
                {
                    results.push_back(it->get());
                }

                return results;
            });
        }

        //! Void version of when_all.  We only support vector containers for void tasks.
        Task<void> when_all(
            std::vector<Task<void>>::iterator begin,
            std::vector<Task<void>>::iterator end)
        {
            return m_defaultQueue.async([begin, end]() -> void
            {
                for (std::vector<Task<void>>::iterator it = begin; it != end; ++it)
                {
                    it->wait();
                }
            });
        }

        //! Return the ASIO service this scheduler owns.
        asio::io_service& getIoService() { return m_ioService; }

    private:
        void workerThread()
        {
            m_ioService.run();
        }

    private:
        //
        // ASIO state
        //
        asio::io_service            m_ioService;
        asio::io_service::work      m_work;

        //
        // Default queues
        //
        ConcurrentQueue             m_defaultQueue;

        //
        // Threads
        //
        std::vector<std::thread>    m_threads;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Utility function to check for power of two value.

    inline constexpr bool isPowerOfTwo(size_t n)
    {
        return ((n & (n - 1)) == 0);
    }

    //------------------------------------------------------------------------------------------------------------------
    //! Base class for all Fifo queues

    class Fifo
    {
    public:
        template <class T>
        void push(T&& item)
        {
            void* p((void *)&item);
            localPush(p);
        }

        template <class T>
        bool pop(T& item)
        {
            void* p((void *)&item);
            return localPop(p);
        }

        virtual bool empty() const = 0;

    protected:
        virtual void localPush(void* p) = 0;
        virtual bool localPop(void* p) = 0;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Implementation of a local fifo

    //! Lock-free multi-threaded ring buffer data structure.
    template <typename T>
    class DataQueue : public Fifo
    {
    public:
        //! Default size of ring buffer is set to 1024 elements.
        static const size_t DefaultSize = 1024;
        static_assert(isPowerOfTwo(DefaultSize));

        //! Constructs a ring buffer with a given number of elements.
        //!
        //! @param  sizeOfBuffer    Number of elements allowed in the buffer.
        DataQueue(size_t sizeOfBuffer = DefaultSize);

        //! Destructor
        ~DataQueue();

        //! Return the number of elements in the ring buffer.
        size_t size() const;

        //! Return true if the queue is empty
        bool empty() const { return size() == 0; }

        //! Return true if the queue is full
        bool full() const;

        //! Push a value on to the ring buffer.
        //!
        //! This function will block this thread until pushing to the queue is successful.  This means that a full
        //! queue with no consumption can cause a deadlock.  Use tryPush() instead to use a non-blocking version.
        //!
        //! @param  value   The value to push on to the buffer.
        void push(T&& value);

        //! Pop a value off the ring buffer.
        //!
        //! Will keep trying to pop off a value from the queue.  This will block until success, or return false
        //! if there are no elements to pop.
        //!
        //! @param  outElem     Output variable that will receive the popped value.
        //! @return             Returns true if successful, or false if there are no items to be popped off.
        bool pop(T& outElem);

        //! Attempt to push a value on to the queue.
        //!
        //! A full buffer could stop a value being pushed.  The consumer will need to pop off values to make room.
        //!
        //! @param  value   The value to push on to the queue.
        //! @return         Returns true if the value was able to be pushed on the queue.
        bool tryPush(T&& value);

    protected:
        void localPush(void *p) override
        {
            T& value = *(T *)p;
            push(std::move(value));
        }

        bool localPop(void *p) override
        {
            T& value = *(T *)p;
            return pop(value);
        }

    private:
        size_t realIndex(size_t index) const
        {
            assert(isPowerOfTwo(m_capacity));
            return index & (m_capacity - 1);
        }

    private:
        T * m_queue;        //!< The fixed array that contains the elements.
        std::atomic<size_t>     m_write;        //!< The position where new elements will be written to.
        std::atomic<size_t>     m_read;         //!< The position where the next element will be popped from.
        std::atomic<size_t>     m_count;        //!< Number of elements in the queue.
        size_t                  m_capacity;     //!< Size of queue's memory allocation (in elements).

                                                //! The maximum position where the next element will be popped from.
                                                //!
                                                //! If m_readMax != m_write, there are reserved but uncomitted writes still pending.  So the thread reading from
                                                //! them will have to wait until the thread writes to it.
        std::atomic<size_t>     m_readMax;

    };

    //------------------------------------------------------------------------------------------------------------------
    //! A port has a reference to a fifo, and helps determine if it's an input or output port.

    class Node;

    class Port
    {
        friend struct NodePair;

    public:
        Port(Node& n, std::function<Fifo*()>& creator, bool isInput)
            : m_node(n)
            , m_fifoCreator(creator)
            , m_isInput(isInput)
        {

        }

        template <typename T>
        bool pop(T& value)
        {
            assert(m_fifos.size() == 1);
            assert(m_isInput);
            return m_fifos[0]->pop(value);
        }

        template <typename T>
        void push(T&& value)
        {
            assert(!m_fifos.empty());
            assert(!m_isInput);

            if (m_fifos.size() == 1)
            {
                m_fifos[0]->push(std::move(value));
            }
            else
            {
                // #todo: We can move one of them and copy the others.
                for (auto& fifo : m_fifos)
                {
                    std::remove_reference<T>::type t = value;
                    fifo->push(std::move(t));
                }
            }
        }

        Fifo* createFifo()
        {
            return m_fifoCreator();
        }

        bool compatibleWith(Port& p)
        {
            return m_type == p.m_type;
        }

        void setFifo(Fifo* fifo)
        {
            assert(m_fifos.empty());
            addFifo(fifo);
        }

        void addFifo(Fifo* fifo)
        {
            m_fifos.push_back(fifo);
        }

        bool hasData() const
        {
            return m_fifos.size() == 1 && !m_fifos[0]->empty();
        }

    private:
        Node & m_node;
        bool                    m_isInput;
        std::vector<Fifo*>      m_fifos;
        std::function<Fifo*()>  m_fifoCreator;
        intptr_t                m_type = 0;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Node I/O port collection
    //
    //! The ports can either be an input or output group

    class Ports
    {
    public:
        Ports(Node& node) : m_node(node) {}

        template <typename T>
        void addPort(std::string&& name);

        template <typename T, typename... PortNamesT>
        void addPort(PortNamesT&&... port);

        virtual bool isInput() const = 0;

        using MaybePort = std::optional<std::reference_wrapper<Port>>;

        MaybePort getPort(const std::string& name)
        {
            auto it = m_ports.find(name);
            return it == m_ports.end() ? MaybePort{} : MaybePort{ it->second };
        }

        MaybePort getPort()
        {
            return (m_ports.size() == 1) ? MaybePort{ m_ports.begin()->second } : MaybePort{};
        }

        Port& operator[] (const std::string& name)
        {
            return m_ports.at(name);
        }

        size_t getNumPorts() const { return m_ports.size(); }

        Node& getNode() { return m_node; }
        const Node& getNode() const { return m_node; }

        bool hasCompleteData() const
        {
            assert(isInput());

            for (const auto& port : m_ports)
            {
                if (!port.second.hasData()) return false;
            }
            return true;
        }

    protected:
        std::map<std::string, Port> m_ports;

    private:
        Node & m_node;
    };

    class InputPorts : public Ports
    {
    public:
        InputPorts(Node& node) : Ports(node) {}

        bool isInput() const override { return true; }

        template <typename T>
        void pop(T& t)
        {
            assert(m_ports.size() == 1);
            m_ports.begin()->second.pop(t);
        }
    };

    class OutputPorts : public Ports
    {
    public:
        OutputPorts(Node& node) : Ports(node) {}

        bool isInput() const override { return false; }

        template <typename T>
        void push(T&& t)
        {
            assert(m_ports.size() == 1);
            m_ports.begin()->second.push(std::forward<T>(t));
        }
    };

    //------------------------------------------------------------------------------------------------------------------
    //! Task graph node

    class Node
    {
        friend class Graph;

    public:
        Port & operator[] (const std::string& name)
        {
            auto inputPort = input.getPort(name);
            if (inputPort)
            {
                return *inputPort;
            }
            else
            {
                auto outputPort = output.getPort(name);
                assert(outputPort);
                return *outputPort;
            }
        }

        Ports::MaybePort getInputPort()
        {
            return input.getPort();
        }

        Ports::MaybePort getOutputPort()
        {
            return output.getPort();
        }

        Ports::MaybePort getPort(const std::string& name)
        {
            auto port = getInputPort(name);
            if (!port)
            {
                port = getOutputPort(name);
            }

            return port;
        }

        Ports::MaybePort getInputPort(const std::string& name)
        {
            return input.getPort(name);
        }

        Ports::MaybePort getOutputPort(const std::string& name)
        {
            return output.getPort(name);
        }

        bool isInitialNode() const
        {
            return input.getNumPorts() == 0 && output.getNumPorts() > 0;
        }

        bool hasCompleteInput() const
        {
            return input.hasCompleteData();
        }

    protected:
        enum class RunState
        {
            Incomplete,
            Complete,
        };
        virtual RunState run() = 0;

    protected:
        InputPorts      input = { *this };
        OutputPorts     output = { *this };
    };

    //------------------------------------------------------------------------------------------------------------------
    //! A connection between two nodes

    struct NodePair
    {
        friend class Graph;

    public:
        NodePair(Port& p1, Port& p2)
            : m_nodeA(&p1.m_node)
            , m_nodeB(&p2.m_node)
            , m_portA(&p1)
            , m_portB(&p2)
            , m_fifo(nullptr)
        {

        }

        Node*   m_nodeA;
        Node*   m_nodeB;
        Port*   m_portA;
        Port*   m_portB;
        Fifo*   m_fifo;
    };

    //------------------------------------------------------------------------------------------------------------------
    //! A graph class that connects Nodes to each other via Fifos

    class Graph final
    {
    public:
        ~Graph();

        Graph& operator+= (NodePair& np);

        //! Execute the entire graph using the given scheduler.
        void exe(Scheduler& scheduler);

    private:
        using NodeSet = std::unordered_set<Node*>;
        NodeSet m_nodeSet;
        std::vector<Fifo*> m_queues;

        NodeSet m_initialNodes;
        NodeSet m_internalNodes;
    };

    //------------------------------------------------------------------------------------------------------------------
    // Graph operators

    //! Create a node pair between two ports.
    inline NodePair operator >> (Port& pa, Port& pb)
    {
        return NodePair{ pa, pb };
    }

    //! Create a node pair between two nodes.
    inline NodePair operator >> (Node& a, Node& b)
    {
        auto p1 = a.getOutputPort();
        auto p2 = b.getInputPort();
        assert(p1);
        assert(p2);

        return *p1 >> *p2;
    }

    //! Create a node pair between a node and a port.
    inline NodePair operator >> (Node& na, Port& pb)
    {
        auto p = na.getOutputPort();
        assert(p);
        return *p >> pb;
    }

    //! Create a node pair between a port and a node.
    inline NodePair operator >> (Port& pa, Node& nb)
    {
        auto p = nb.getInputPort();
        assert(p);
        return pa >> *p;
    }

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------

} // namespace cc

  //----------------------------------------------------------------------------------------------------------------------
  //----------------------------------------------------------------------------------------------------------------------
  // I M P L E M E N T A T I O N
  //----------------------------------------------------------------------------------------------------------------------
  //----------------------------------------------------------------------------------------------------------------------

namespace cc
{

    //------------------------------------------------------------------------------------------------------------------

    inline void ConcurrentQueue::pushTask(const std::shared_ptr<TaskBase>& task)
    {
        getScheduler().getIoService().post(std::bind(&TaskBase::execute, task.get()));
    }

    //------------------------------------------------------------------------------------------------------------------

    inline SerialQueue::SerialQueue(Scheduler& scheduler)
        : Queue(scheduler)
        , m_strand(scheduler.getIoService())
    {}

    //------------------------------------------------------------------------------------------------------------------

    inline void SerialQueue::pushTask(const std::shared_ptr<TaskBase>& task)
    {
        m_strand.post(std::bind(&TaskBase::execute, task.get()));
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        DataQueue<T>::DataQueue(size_t sizeOfBuffer /* = DefaultSize */)
        : m_queue(new T[sizeOfBuffer + 1])
        , m_write(0)
        , m_read(0)
        , m_count(0)
        , m_capacity(sizeOfBuffer)
        , m_readMax(0)
    {
        assert(isPowerOfTwo(sizeOfBuffer));
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        DataQueue<T>::~DataQueue()
    {
        delete[] m_queue;
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        size_t DataQueue<T>::size() const
    {
        return m_count;
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        bool DataQueue<T>::full() const
    {
        return m_count == (m_capacity - 1);
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        bool DataQueue<T>::tryPush(T&& value)
    {
        size_t currentWriteIndex = m_write;
        if (realIndex(currentWriteIndex + 1) == realIndex(m_read))
        {
            // The queue is full.
            return false;
        }

        // Attempt to reserve the write space using lock-free techniques.
        if (!m_write.compare_exchange_strong(currentWriteIndex, (currentWriteIndex + 1))) return false;

        m_queue[realIndex(currentWriteIndex)] = std::move(value);

        // Update the final read point.
        while (!m_readMax.compare_exchange_weak(currentWriteIndex, (currentWriteIndex + 1)));
        m_count.fetch_add(1);

        return true;
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        bool DataQueue<T>::pop(T& outElem)
    {
        do
        {
            size_t currentReadIndex = m_read;

            if (realIndex(currentReadIndex) == realIndex(m_readMax))
            {
                // The queue is empty, or a producer thread has allocated space but has not yet committed it.
                return false;
            }

            // Retrieve the data from the queue
            T* p = &m_queue[realIndex(currentReadIndex)];

            if (m_read.compare_exchange_strong(currentReadIndex, (currentReadIndex + 1)))
            {
                // The value was retrieved atomically from the queue.
                m_count.fetch_sub(1);
                outElem = std::move(*p);
                return true;
            }
        } while (1);
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T> inline
        void DataQueue<T>::push(T&& value)
    {
        while (!tryPush(std::move(value)));
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T, typename... PortNamesT>
    struct PortHelper
    {

    };

    template <typename T>
    struct PortHelper<T>
    {
        static void addPort(Ports& port)
        {

        }
    };

    template <typename T, typename PortNameT, typename... PortNamesT>
    struct PortHelper<T, PortNameT, PortNamesT...>
    {
        static void addPort(Ports& ports, PortNameT&& portName, PortNamesT&&... portNames)
        {
            ports.template addPort<T>(std::string(portName));
            PortHelper<T, PortNamesT...>::addPort(ports, std::forward<PortNamesT>(portNames)...);
        }
    };

    //------------------------------------------------------------------------------------------------------------------

    template <typename T, typename... PortNamesT> inline
        void Ports::addPort(PortNamesT&&... portNames)
    {
        PortHelper<T, PortNamesT...>::addPort(*this, std::forward<PortNamesT>(portNames)...);
    }

    //------------------------------------------------------------------------------------------------------------------

    template <typename T>
    void Ports::addPort(std::string&& name)
    {
        static auto createFunc = []() -> Fifo* {
            return new DataQueue<T>(kDefaultDataQueueSize);
        };

        m_ports.emplace(std::forward<std::string>(name), Port(getNode(), std::function<Fifo*()>(createFunc), isInput()));
    }

    //------------------------------------------------------------------------------------------------------------------

    inline Graph::~Graph()
    {
        for (auto& queue : m_queues)
        {
            delete queue;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    inline Graph& Graph::operator+= (NodePair& np)
    {
        m_nodeSet.emplace(np.m_nodeA);
        m_nodeSet.emplace(np.m_nodeB);

        assert(np.m_portA->compatibleWith(*np.m_portB));

        Fifo* fifo = np.m_portA->createFifo();
        m_queues.push_back(fifo);
        np.m_portA->addFifo(fifo);
        np.m_portB->setFifo(fifo);

        if (np.m_nodeA->isInitialNode())
        {
            m_initialNodes.emplace(np.m_nodeA);
        }
        else
        {
            m_internalNodes.emplace(np.m_nodeA);
        }
        m_internalNodes.emplace(np.m_nodeB);

        return *this;
    }

    //------------------------------------------------------------------------------------------------------------------

    inline void Graph::exe(Scheduler& scheduler)
    {
        using TrackedNode = std::pair<Node::RunState, Node*>;
        using TrackedInitialNodes = std::vector<TrackedNode>;

        TrackedInitialNodes startNodes;
        std::transform(m_initialNodes.begin(), m_initialNodes.end(), inserter(startNodes, startNodes.begin()), [](Node* n) -> TrackedNode {
            return std::make_pair(Node::RunState::Incomplete, n);
        });

        int numNodesRan = 0;
        while (numNodesRan ||
            any_of(startNodes.begin(), startNodes.end(), [](const TrackedNode& n) { return n.first == Node::RunState::Incomplete; }))
        {
            numNodesRan = 0;

            //
            // Step 1 - run the output-only kernels to kick-start the process
            //
            for (TrackedNode& node : startNodes)
            {
                if (node.first == Node::RunState::Incomplete)
                {
                    node.first = node.second->run();
                }
            }

            //
            // Step 2 - run through all the internal nodes.  If any have input data then run the kernel.
            //
            for (Node* node : m_internalNodes)
            {
                if (node->hasCompleteInput())
                {
                    ++numNodesRan;
                    node->run();
                }
            }
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------

} // namespace cc

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
