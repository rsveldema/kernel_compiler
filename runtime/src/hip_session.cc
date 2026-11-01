#include <hip_session.hpp>
#include <thread>
HipSession::HipSession(bool,const char*preferred){int count=0;if(hipGetDeviceCount(&count)!=hipSuccess||count==0)return;for(int i=0;i<count;++i){hipDeviceProp_t p{};check_hip(hipGetDeviceProperties(&p,i),"hipGetDeviceProperties");if(!preferred||std::string_view(p.name).find(preferred)!=std::string_view::npos){device_=i;max_allocation_=p.totalGlobalMem;break;}}check_hip(hipSetDevice(device_),"hipSetDevice");has_device_=true;const size_t n=std::clamp<size_t>(std::thread::hardware_concurrency(),1,16);for(size_t i=0;i<n;++i)queues_.push_back(std::make_unique<HipQueue>(i));}
HipComputeKernel::HipComputeKernel(HipSession&,const std::string&path,const std::string&symbol){check_hip(hipModuleLoad(&module_,path.c_str()),"hipModuleLoad");check_hip(hipModuleGetFunction(&function_,module_,symbol.c_str()),"hipModuleGetFunction");}
HipComputeKernel::~HipComputeKernel(){if(module_)(void)hipModuleUnload(module_);}
void HipComputeKernel::launch(HipQueue&q,HipDimension g,HipDimension b,void**args)const{check_hip(hipModuleLaunchKernel(function_,g.x,g.y,g.z,b.x,b.y,b.z,0,q.stream(),args,nullptr),"hipModuleLaunchKernel");}
