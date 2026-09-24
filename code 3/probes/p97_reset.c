// P97d：最小 ACL 复位程序 —— 只看三个调用的返回码，不做任何计算。
// 动机：cube 臂被前面几次挂死/SIGKILL 毒化后，同一份字节的 p6 连 10 遍 rc=2；
// 需要一个"不重启容器"的恢复手段。0 = ACL_SUCCESS。
#include <acl/acl.h>
#include <stdio.h>

int main(void)
{
    printf("aclInit          = %d\n", (int)aclInit(nullptr));
    printf("aclrtSetDevice(0)= %d\n", (int)aclrtSetDevice(0));
    printf("aclrtResetDevice = %d\n", (int)aclrtResetDevice(0));
    printf("re-setDevice     = %d\n", (int)aclrtSetDevice(0));
    printf("reset(2nd)       = %d\n", (int)aclrtResetDevice(0));
    return 0;
}
