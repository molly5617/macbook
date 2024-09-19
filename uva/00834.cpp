#include <iostream>
using namespace std;
#include <vector>
int main()
{
    int a, b;
    while (cin >> a >> b)
    {
        vector<int> ans;
        while (b != 0)
        {
            ans.push_back(a / b);
            a = a % b;
            int temp = a;
            a = b;
            b = temp;
        }
        cout << "[" << ans[0] << ";" << ans[1];
        for (int i = 2; i < ans.size(); i++)
        {
            cout << "," << ans[i];
        }
        cout << "]" << endl;
    }
}