import subprocess
import editdistance


def evaluate(image_path1, image_path2):

    subprocess.call('tesseract '+image_path1+' '+'result1', shell=True)
    subprocess.call('tesseract '+image_path2 + ' ' + 'result2', shell=True)
    with open('result1.txt', 'r') as file1:
        text1 = file1.read()
    with open('result2.txt', 'r') as file2:
        text2 = file2.read()

    return editdistance.eval(text1, text2)
