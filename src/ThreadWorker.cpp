// -*- C++ -*-  ---------------------------------------------------------

#include "ThreadWorker.h"

#include <boost/bind.hpp>
#include <string>
#include <sstream>
#include <iostream>

using namespace boost;
using namespace std;

namespace threadworker{

/*
Constructing a ThreadWorker creates the worker thread 
*/
ThreadWorker::ThreadWorker() : m_exit(false), m_work_to_do(false), m_last_error(ThreadWorker::all_success)
{
  m_tagged_file = "n/a";
  m_tagged_line = 0;
  m_thread.reset(new thread(bind(&ThreadWorker::performWorkLoop, this)));
}


/*! Shuts down the worker thread
*/
ThreadWorker::~ThreadWorker()
{
  // set the exit condition
  {
    mutex::scoped_lock lock(m_mutex);
    m_work_to_do = true;
    m_exit = true;
  }

  // notify the thread there is work to do
  m_cond_work_to_do.notify_one();

  // join with the thread
  m_thread->join();
}


/*! \param func Function call to execute in the worker thread

call() executes a CUDA call to in a worker thread. Any function
with any arguments can be passed in to be queued using boost::bind.
Examples:
\code
gpu.call(bind(function, arg1, arg2, arg3, ...));
gpu.call(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost));
gpu.call(bind(cudaThreadSynchronize));
\endcode
The only requirement is that the function returns a cudaError_t. Since every
single CUDA Runtime API function does so, you can call any Runtime API function.
You can call any custom functions too, as long as you return a cudaError_t representing
the error of any CUDA functions called within. This is typical in kernel
driver functions. For example, a .cu file might contain:
\code
__global__ void kernel() { ... }
cudaError_t kernel_driver()
{
kernel<<<blocks, threads>>>();
#ifdef NDEBUG
return cudaSuccess;
#else
cudaThreadSynchronize();
return cudaGetLastError();
#endif
}
\endcode
It is recommended to just return cudaSuccess in release builds to keep the asynchronous
call stream going with no cudaThreadSynchronize() overheads.

call() ensures that \a func has been executed before it returns. This is
desired behavior, most of the time. For calling kernels or other asynchronous
CUDA functions, use callAsync(), but read the warnings in it's documentation
carefully and understand what you are doing. Why have callAsync() at all?
The original purpose for designing ThreadWorker is to allow execution on 
multiple GPUs simultaneously which can only be done with asynchronous calls.

An exception will be thrown if the CUDA call returns anything other than
cudaSuccess.
*/
  void ThreadWorker::call(const boost::function< ThreadWorker::error_t (void) > &func, int device)
{
  // this mutex lock is to prevent multiple threads from making
  // simultaneous calls. Thus, they can depend on the exception
  // thrown to exactly be the error from their call and not some
  // race condition from another thread
  // making ThreadWorker calls to a single ThreadWorker from multiple threads 
  // still isn't supported
  mutex::scoped_lock lock(m_call_mutex);

#ifdef HAVE_CUDA
  if(device != -1)
    {
      callAsync(boost::bind(cudaSetDevice,device));
    }
#endif

  // call and then sync
  callAsync(func);
  sync();
}


/*! \param func Function to execute inside the worker thread

callAsync is like call(), but  returns immeadiately after entering \a func into the queue. 
The worker thread will eventually get around to running it. Multiple contiguous
calls to callAsync() will result in potentially many function calls 
being queued before any run.

\warning There are many potential race conditions when using callAsync().
For instance, consider the following calls:
\code
gpu.callAsync(bind(cudaMalloc(&d_array, n_bytes)));
gpu.callAsync(bind(cudaMemcpy(d_array, h_array, n_bytes, cudaMemcpyHostToDevice)));
\endcode
In this code sequence, the memcpy async call may be created before d_array is assigned
by the malloc call leading to an invalid d_array in the memcpy. Similar race conditions
can show up with device to host memcpys. These types of race conditions can be very hard to
debug, so use callAsync() with caution. Primarily, callAsync() should only be used to call
cuda functions that are asynchronous normally. If you must use callAsync() on a synchronous
cuda function (one valid use is doing a memcpy to/from 2 GPUs simultaneously), be
\b absolutely sure to call sync() before attempting to use the results of the call.

\warning Arguments that are passed into the function call by bind are put into a queue.
They may have a lifetime longer than that of the caller. If any function performs a
callAsync and uses pointers to stack variables in the call, sync() \b must be called
at the end of the function to avoid problems. Similarly, sync() must be called in the
destructor of any class that passes pointers to member variables into callAsync().

The best practice to avoid problems is to always call sync() at the end of any function 
that uses callAsync().
*/
  void ThreadWorker::callAsync(const boost::function< ThreadWorker::error_t (void) > &func,int device)
{
  // add the function object to the queue
  {
    mutex::scoped_lock lock(m_mutex);
#ifdef HAVE_CUDA
   if(device != -1)
    {
      m_work_queue.push_back(boost::bind(cudaSetDevice,device));
    }
#endif

    m_work_queue.push_back(func);
    m_work_to_do = true;
  }

  // notify the threads there is work to do
  m_cond_work_to_do.notify_one();
}

/*! Call sync() to synchronize the master thread with the worker thread.
After a call to sync() returns, it is guarunteed that all previous
queued calls (via callAsync()) have been called in the worker thread. 
For the CUDA enabled version of the library this will also call
cudaThreadSynchronize()

sync() will throw an exception if any of the queued calls resulted in
a return value not equal to cudaSuccess.
*/
void ThreadWorker::sync()
{
#ifdef HAVE_CUDA
  callAsync(boost::bind(cudaDeviceSynchronize));
#endif

  // wait on the work done signal
  // wait on the work done signal
  mutex::scoped_lock lock(m_mutex);
  while (m_work_to_do)
    m_cond_work_done.wait(lock);


  // if there was an error
  if (m_last_error & all_error)
  {
    if(m_last_error & cuda_error)
    {
#ifdef HAVE_CUDA
    // build the exception
    //cerr << endl << "***Error! " << string(cudaGetErrorString(m_last_error)) << " after " << m_tagged_file << ":" << m_tagged_line << endl << endl;
      cerr << endl << "***Error! " << " after " << m_tagged_file << ":" << m_tagged_line << endl << endl;
    //runtime_error error("CUDA Error");
#endif
    }
    else
    {
    // build the exception
    cerr << endl << "***Error! " << " after " << m_tagged_file << ":" << m_tagged_line << endl << endl;
    }
    runtime_error error("Error");

    // reset the error value so that it doesn't propagate to continued calls
    m_last_error = all_success;

    // throw
    throw(error);
  }	
}


/*! \param file Current file of source code
\param line Current line of source code

This is intended to be called worker.setTag(__FILE__, __LINE__). When reporting errors,
the last file and line tagged will be printed to help identify where the error occured.
*/
void ThreadWorker::setTag(const std::string &file, unsigned int line)
{
  m_tagged_file = file;
  m_tagged_line = line;
}

/*! \internal
The worker thread spawns a loop that continusously checks the condition variable
m_cond_work_to_do. As soon as it is signaled that there is work to do with
m_work_to_do, it processes all queued calls. After all calls are made,
m_work_to_do is set to false and m_cond_work_done is notified for anyone 
interested (namely, sync()). During the work, m_exit is also checked. If m_exit
is true, then the worker thread exits.
*/
void ThreadWorker::performWorkLoop()
{
  bool working = true;

  // temporary queue to ping-pong with the m_work_queue
  // this is done so that jobs can be added to m_work_queue while
  // the worker thread is emptying pong_queue
  deque< boost::function< error_t (void) > > pong_queue;

  while (working)
  {
    // aquire the lock and wait until there is work to do
    {
      mutex::scoped_lock lock(m_mutex);
      while (!m_work_to_do)
        m_cond_work_to_do.wait(lock);

      // check for the exit condition
      if (m_exit)
        working = false;

      // ping-pong the queues
      pong_queue.swap(m_work_queue);
    }

    // track any error that occurs in this queue
    error_t error = all_success;

    // execute any functions in the queue
    while (!pong_queue.empty())
    {
      // cout << " at " << m_tagged_file << ":" << m_tagged_line << endl;
      error_t tmp_error = pong_queue.front()();

      // update error only if it is cudaSuccess
      // this is done so that any error that occurs will propagate through
      // to the next sync()
      if (error & all_success) // Bitwise or
        error = tmp_error;

      pong_queue.pop_front();
    }

    // reaquire the lock so we can update m_last_error and 
    // notify that we are done
    {
      mutex::scoped_lock lock(m_mutex);

      // update m_last_error only if it is cudaSuccess
      // this is done so that any error that occurs will propagate through
      // to the next sync()
      if (m_last_error & all_success)
        m_last_error = error;

      // notify that we have emptied the queue, but only if the queue is actually empty
      // (call_async() may have added something to the queue while we were executing above)
      if (m_work_queue.empty())
      {
        m_work_to_do = false;
        m_cond_work_done.notify_all();
      }
    }
  }
}

} // threadworker

