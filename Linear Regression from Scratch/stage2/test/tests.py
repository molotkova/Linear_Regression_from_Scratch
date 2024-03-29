import re

import numpy as np
from hstest import StageTest, CheckResult, dynamic_test, TestedProgram


def get_number(string):
    return list(map(float, re.findall(r'-*\d*\.\d+|-*\d+', string)))


answer = np.array([35.35993478, 41.13594995, 45.19749381, 51.16062856, 52.99980512, 60.91799349, 61.01596736])


class LinearRegression(StageTest):

    @dynamic_test()
    def test_1(self):
        t = TestedProgram()
        reply = t.start()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed. Print output in the right format.")

        if '[' not in reply or ']' not in reply or ',' in reply:
            return CheckResult.wrong("Print output as numpy array")

        reply = get_number(reply)

        if len(reply) != 7:
            return CheckResult.wrong(f"y should contain 7 values, found {len(reply)}")

        for reply_coef, answer_coef in zip(reply, answer):
            # 2% error is allowed
            error = answer_coef * 0.02
            if not answer_coef - error < reply_coef < answer_coef + error:
                return CheckResult.wrong(
                    f"Incorrect y array. Check your fit_intercept=False and predict method implementations")

        return CheckResult.correct()


if __name__ == '__main__':
    LinearRegression().run_tests()