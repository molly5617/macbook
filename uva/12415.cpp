#include <iostream>
using namespace std;
#include <string>
#include <vector>
#include <algorithm>
int main()
{
    int n, target, p;
    cin >> n;
    while (n--)
    {
        cin >> target;
        cin >> p;
        vector<int> v;
        for (int i = 0; i < p; i++)
        {
            int temp;
            cin >> temp;
            v.push_back(temp);
        }
        vector<int> dp(target + 1, 0);
        dp[0] = 0;
        for (int i = 0; i < p; i++)
        {
            for (int j = target; j > 0; j--)
            {
                if (j >= v[i] && j <= v[i] + dp[j - v[i]])
                {
                    dp[j] = max(dp[j], v[i] + dp[j - v[i]]);
                }
            }
        }
        if (dp[target] == target)
            cout << "YES" << endl;
        else
            cout << "NO" << endl;
    }
}