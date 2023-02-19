./build.ps1
cd ./bin
try {
    ./main.exe
}
finally {
    cd ..
}