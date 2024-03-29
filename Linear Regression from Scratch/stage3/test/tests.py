import re

from hstest import StageTest, CheckResult, dynamic_test, TestedProgram


def get_number(string):
    return list(map(float, re.findall(r'-*\d*\.\d+|-*\d+', string)))


class LinearRegression(StageTest):

    @dynamic_test()
    def test_1(self):
        t = TestedProgram()
        reply = t.start()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed. Print output in the right format.")

        if '{' not in reply or '}' not in reply or ":" not in reply or "," not in reply:
            return CheckResult.wrong("Print output in dictionary format")

        if 'array' not in reply:
            return CheckResult.wrong("Return coefficient(s) in numpy array")

        if reply.count(',') != 4 or reply.count(':') != 4 or reply.count('}') != reply.count('{'):
            return CheckResult.wrong('The dictionary output is not properly constructed.')

        output = reply.replace("{", "").replace("}", "").lower().split(", '")

        if len(output) != 4:
            return CheckResult.wrong(f"No of items in dictionary should be 4, {len(output)} present")

        output = [j.replace("'", "") for j in output]
        output1, output2, output3, output4 = output

        name1, answer1 = output1.strip().split(':')
        name2, answer2 = output2.strip().split(':')
        name3, answer3 = output3.strip().split(':')
        name4, answer4 = output4.strip().split(':')

        answers = {
            name1.strip(): answer1, name2.strip(): answer2,
            name3.strip(): answer3, name4.strip(): answer4}

        intercept = answers.get('intercept', '0000000')
        coefficient = answers.get('coefficient', '0000000')
        coefficient = re.sub('array', '', coefficient)
        r2 = answers.get('r2', '0000000')
        rmse = answers.get('rmse', '0000000')

        if intercept == '0000000' or coefficient == '0000000' or len(intercept) == 0 or len(coefficient) == 0:
            return CheckResult.wrong("Print values for both Intercept and Coefficient")

        if r2 == '0000000' or rmse == '0000000' or len(r2) == 0 or len(rmse) == 0:
            return CheckResult.wrong("Print values for both R2 and RMSE")

        intercept = get_number(intercept)
        if len(intercept) != 1:
            return CheckResult.wrong(f"Intercept should contain a single value, found {len(intercept)}")
        intercept = intercept[0]
        if not 18.6 < intercept < 18.8:
            return CheckResult.wrong("Wrong value for Intercept")

        coefficient = get_number(coefficient)
        if len(coefficient) != 2:
            return CheckResult.wrong(f"Coefficient should contain two values, found {len(coefficient)}")
        if not -3.3 < coefficient[0] < -3.1:
            return CheckResult.wrong("Wrong value for beta1")
        if not 0.5 < coefficient[1] < 0.7:
            return CheckResult.wrong("Wrong value for beta2")

        r2 = get_number(r2)
        if len(r2) != 1:
            return CheckResult.wrong(f"R2 should contain a single value, found {len(r2)}")
        r2 = r2[0]
        if not 0.7 < r2 < 0.9:
            return CheckResult.wrong("Wrong value for R2 score")

        rmse = get_number(rmse)
        if len(rmse) != 1:
            return CheckResult.wrong(f"RMSE should contain a single value, found {len(rmse)}")
        rmse = rmse[0]
        if not 1.6 < rmse < 1.8:
            return CheckResult.wrong("Wrong value for RMSE score. Did you take the square root of mean squared error?")

        return CheckResult.correct()


if __name__ == '__main__':
    LinearRegression().run_tests()