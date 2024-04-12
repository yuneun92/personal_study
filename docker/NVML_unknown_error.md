1. 컨테이너 밖에서 `nvidia-smi` 했을 때는
![image](https://github.com/yuneun92/personal_study/assets/101092482/eafbae88-9bef-4518-95a7-295f4adc54ab)
이렇게 잘 잡히는데,

`nvcc -V`로는 버전이 확인되지 않고(`Command 'nvcc' not found, but can be installed with: sudo apt install nvidia-cuda-toolkit`)

컨테이너에 들어가서 `nvidia-smi` 치면 
`Failed to initialize NVML: Unknown Error` 이런 에러가 뜸 -- 해결 방법 찾는 중
