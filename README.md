# sgx-mnist
```bash
docker build -t sgx-mnist .
docker run -d --rm --device /dev/isgx --device /dev/mei0 --name test-sgx sgx-mnist
docker exec -t -i test-sgx bash run.sh
docker stop test-sgx
```
