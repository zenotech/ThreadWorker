// -*- C++ -*-

/*
This is a modified version of GPUWorker from HOOMD Blue
Enables the use of the master worker concept for GPU or host tasking
Copyright (c) 2012 Zenotech Ltd
All right reserved
*/

/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

/*! \file ThreadWorker.h
\brief Defines the ThreadWorker class
*/

#pragma once

#include <cstdio>
#include <deque>
#include <stdexcept>
#include <memory>
#include <functional>
#include <thread>
#include <condition_variable>

#ifdef WIN32
#  if defined ThreadWorker_EXPORTS || defined ThreadWorkerCUDA_EXPORTS
#    define THREADWORKER_EXPORT __declspec(dllexport)
#  else
#    define THREADWORKER_EXPORT __declspec(dllimport)
#  endif
#else
#  define THREADWORKER_EXPORT
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace threadworker
{

//! Implements a worker thread controlling a single GPU
/*! CUDA requires one thread per GPU in multiple GPU code. It is not always
convenient to write multiple-threaded code where all threads are peers.
Sometimes, a master/slave approach can be the simplest and quickest to write.

ThreadWorker provides the underlying worker threads that a master/slave
approach needs to execute on multiple GPUs. It is designed so that
a \b single thread can own multiple ThreadWorkers, each of whom execute on
their own GPU. The master thread can call any CUDA function on that GPU
by passing a bound std::function into call() or callAsync(). Internally, these
calls are executed inside the worker thread so that they all share the same
CUDA context.

On construction, a ThreadWorker is automatically associated with a device. You
pass in an integer device number which is used to call cudaSetDevice()
in the worker thread.

After the ThreadWorker is constructed, you can make calls on the GPU
by submitting them with call(). To queue calls, use callAsync(), but
please read carefully and understand the race condition warnings before
using callAsync(). sync() can be used to synchronize the master thread
with the worker thread. If any called GPU function returns an error,
call() (or the sync() after a callAsync()) will throw a std::runtime_error.

To share a single ThreadWorker with multiple objects, use boost::shared_ptr.
\code
boost::shared_ptr<ThreadWorker> gpu(new ThreadWorker(dev));
gpu->call(whatever...)
SomeClass cls(gpu);
// now cls can use gpu to execute in the same worker thread as everybody else
\endcode

\warning A single ThreadWorker is intended to be used by a \b single master thread
(though master threads can use multiple ThreadWorkers). If a single ThreadWorker is
shared amoung multiple threads then ther \e should not be any horrible consequences.
All tasks will still be exected in the order in which they
are recieved, but sync() becomes ill-defined (how can one synchronize with a worker that
may be receiving commands from another master thread?) and consequently all synchronous
calls via call() \b may not actually be synchronous leading to weird race conditions for the
caller. Then againm calls via call() \b might work due to the inclusion of a mutex lock:
still, multiple threads calling a single ThreadWorker is an untested configuration.
Use at your own risk.

\note ThreadWorker works in both Linux and Windows (tested with VS2005). However,
in Windows, you need to define BOOST_BIND_ENABLE_STDCALL in your project options
in order to be able to call CUDA runtime API functions with boost::bind.

\ingroup utils
*/


class THREADWORKER_EXPORT ThreadWorker
{
public:
  /*
#ifdef HAVE_CUDA
  typedef cudaError_t error_t;
  static error_t success;
#else
  typedef int error_t;
  enum {success = 1};
#endif
*/
  typedef int error_t;
  enum{
    host_success = 0x1,
    cuda_success = 0x2,
    all_success  = 0x3,
    host_error   = 0x4,
    cuda_error   = 0x8,
    all_error    = 0xC,
  };

  //! Creates a worker thread
  ThreadWorker();

  //! Destructor
  ~ThreadWorker();

  //! Makes a synchronous function call executed by the worker thread
  void call(const std::function< error_t (void) > &func, int device);

  //! Queues an asynchronous function call to be executed by the worker thread
  void callAsync(const std::function< error_t (void) > &func, int device);

  //! Blocks the calling thread until all queued calls have been executed
  void sync(int device);

  //! Tag the current location in the code
  void setTag(const std::string &file, unsigned int line);

private:
  //! Flag to indicate the worker thread is to exit
  volatile bool m_exit;

  //! Flag to indicate there is work to do
  volatile bool m_work_to_do;

  //! Error from last cuda call
  error_t m_last_error;

  //! Tagged file
  std::string m_tagged_file;

  //! Tagged line
  unsigned int m_tagged_line;

  //! The queue of function calls to make
  std::deque< std::function< error_t (void) > > m_work_queue;

  //! Mutex for accessing m_exit, m_work_queue, m_work_to_do, and m_last_error
  std::mutex m_mutex;

  //! Mutex for syncing after every operation
  std::mutex m_call_mutex;

  //! Condition variable to signal m_work_to_do = true
  std::condition_variable m_cond_work_to_do;

  //! Condition variable to signal m_work_to_do = false (work is complete)
  std::condition_variable m_cond_work_done;

  //! Thread
  std::unique_ptr<std::thread> m_thread;

  //! Worker thread loop
  void performWorkLoop();
};


namespace detail
{
#ifdef HAVE_CUDA
/*
 * Helper function to trap CUDA errors in the correct format
 */
  inline ThreadWorker::error_t cudaFunc(const std::function<cudaError_t (void) > &func)
  {
    printf("Calling");
    fflush(stdout);
    cudaError_t cudaErr = func();
    ThreadWorker::error_t success = (cudaErr == cudaSuccess ? ThreadWorker::cuda_success : ThreadWorker::cuda_error);
    return success;
  }
#endif
}

} // threadworker
