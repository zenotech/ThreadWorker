// -*- C++ -*-  ---------------------------------------------------------


#include <iostream>

#include "ThreadWorker.h"

ThreadWorker::error_t testFunc0(){

  return ThreadWorker::host_success;
}

ThreadWorker::error_t testFunc1(int a){
  std::cout << "testFunc1: a=" << a << std::endl;
  return ThreadWorker::host_success;
}

ThreadWorker::error_t testFunc2(int a, int &b){
  std::cout << "testFunc2: a=" << a << " b=" << b << std::endl;
  b++;
  return ThreadWorker::host_success;
}


int main()
{

  ThreadWorker worker[4];

  /*
  worker[0].setTag(__FILE__, __LINE__);
  worker[0].call(boost::bind(testFunc0));

  worker[0].callAsync(boost::bind(testFunc0));
  worker[0].sync();
  int a=1;
  worker[0].call(boost::bind(testFunc1,a));
  int b=2;
  worker[0].call(boost::bind(testFunc2,a,boost::ref(b)));
  if(b != 3)
    return EXIT_FAILURE;
*/
  return EXIT_SUCCESS;
}
