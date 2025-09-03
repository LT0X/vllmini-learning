import uvicorn
from .server import app
import argparse
from vllmini.modle_type import IsSpdecode


if __name__ == "__main__":

    #添加参数启动，选用投机解码模式
    parser = argparse.ArgumentParser(description='工作模式')
    parser.add_argument('-spdecode', '--SpeculativeDecoding',
                        action='store_true', help='投机解码')
    isSpdecode =  parser.parse_args().SpeculativeDecoding
    print("isSpdecode的参数是什么 "+str(isSpdecode))
    uvicorn.run(app, host="0.0.0.0", port=8000)


#curl -X POST "http://localhost:8000/generatePro" -H "Content-Type: application/json" -d '{"prompt": "你还记得", "max_length": 8}'
#curl "http://localhost:8000/resultPro/1756626723990544"


