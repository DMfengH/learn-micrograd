#pragma once
#include <chrono>
#include <iostream>

//  这个类可能作为一个命名空间更合适。实例化一个对象很多余

namespace Logger{
  enum logLevel{ErrorLevel=0, WarnLevel, InfoLevel};

  void setLogLevel(logLevel logLevel);
  logLevel getLogLevel();

  template<typename... Args> 
  void error(Args... args){
    log(ErrorLevel, "ERROR", args...);
  }

  template<typename... Args> 
  void warn(Args... args){
    log(WarnLevel, "WARN", args...);
  }

  template<typename... Args> 
  void info(Args... args){
    log(InfoLevel, "INFO", args...);
  }

  template<typename... Args>
  void log(logLevel loglevel, const char* loglevelName, Args... args){
    if(getLogLevel() >= loglevel){
      std::cout << "[ " << loglevelName << " ] ";
      ((std::cout << args << " "), ...);
      std::cout << std::endl;
    }
  }
}


using namespace std::literals::chrono_literals;
class Timer{
public:
  Timer(): m_name("NONAME"){
    start = std::chrono::high_resolution_clock::now();
  }
  Timer(const char* name): m_name(name){
    start = std::chrono::high_resolution_clock::now();
  }
  
  ~Timer(){
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    auto ms = duration.count()*1000;
    std::cout << "[ TIMER-" << m_name << " ] " <<  ms << "(ms)" << std::endl; 
  }

private:
  std::chrono::_V2::system_clock::time_point start;
  const char* m_name;

};