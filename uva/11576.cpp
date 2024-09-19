#include <iostream>
using namespace std;
#include <vector>
#include <string>
#include <algorithm>
int main()
{
    int n;
    cin >> n;
    while (n--)
    {
        int len;
        cin >> len;
        int k;
        cin >> k;
        vector<string> v;
        v.clear();
        for (int i = 0; i < k; i++)
        {
            string str;
            cin >> str;
            v.push_back(str);
        }
        int res = len;
        for (int i = 1; i < k; i++)
        {
            int max = 0;
            res += len;
            for (int j = 0; j < len; j++)
            {
                if (v[i - 1].substr(j, len - j) == v[i].substr(0, len - j))
                {

                    res = res - (len - j);
                    break;
                }
            }
        }
        cout << res << endl;
    }
}