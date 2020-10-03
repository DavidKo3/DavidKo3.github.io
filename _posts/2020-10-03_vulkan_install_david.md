```
layout: post
title:  "2020-10-03_vulkan_install_david"
date:   2020-10-03 12:11:35 +0900
categories: jekyll update
```

# Vulkan 설치

## 1. Vulkan 설치 환경

Ubuntu 18.04 이상에서는 apt 방식으로 설치 할 수 있지만 현재 노트북 환경(16.04)에서는 직접 홈페이지에서 sdk파일을 다운로드 받아 작업합니다.

IDE 는  code-lite  를 사용합니다.

## 2. 설치 순서

다운로드 받은 파일에서 pkg-config 파일들을 확인합니다.

현재 폴더 주소는 다음과 같습니다.

![vulkan_pkg-config_location](/assets/vulkan_related/1_vulkan_pkg-config_location.png)

다음으로  vulkan.pc 파일을 열어 prefix, exec_prefix를 현재 설치된 주소로 변경합니다.

![](/assets/vulkan_related/2_vulakn_pc_fix.png)

cmd 환경에서 `pkg-config --cflags --libs vulkan` 명령을 실행하여 include 파일 위치와 library 파일 위치가 제대로 설정되어 있는지 확인 합니다.

 ![3_check_vulkan_pkg-config](/home/davidk/Desktop/vulkan_related/3_check_vulkan_pkg-config.png)



code lite를 실행하고  vulkan 프로젝트를 만들고 settings-> linker option에서 `pkg-config --cflags --libs vulkan` 을 추가합니다.

 ![](/assets/vulkan_related/4_codelite_vulkan_pkg_config.png)



마지막으로 예제 실행 파일이 잘 실행되는지 확인합니다.

![](/assets/vulkan_related/5_codelite_vulkan_build_check.png)

