#include "utilities.h"

// 在命名空间中定义一个全局变量。
namespace Logger{
  namespace{
    logLevel LOGLEVEL = InfoLevel;
  }

logLevel getLogLevel(){
  return LOGLEVEL;
}

void setLogLevel(logLevel logLevel){
  LOGLEVEL = logLevel;
}
}

