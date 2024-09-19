#include <iostream>
#include <vector>
#include<algorithm>
using namespace std;
string ans;
vector<char> tmp;
vector<string> res;
int now = 0;

void dfs(string &str, int start, vector<bool> &track)
{
    if (tmp.size() == str.size())
    {
        ans = "";
        for (int j = 0; j < tmp.size(); j++)
        {
            ans += tmp[j];
        }
        res.push_back(ans);
        return;
    }
    for (int i = 0; i < str.size(); i++)
    {
        if (track[i] == true)
            continue;

        track[i] = true;
        tmp.push_back(str[i]);
        dfs(str, i, track);
        tmp.pop_back();
        track[i] = false;
    }
}
int main()
{
    int n;
    cin >> n;
    int pos;
    string str;
    for (int i = 0; i < n; i++)
    {
        cin >> str;
        cin >> pos;
        tmp.clear();
        ans = "";
        res.clear();
        sort(str.begin(), str.end());
        vector<bool> track(str.size(), false);
        dfs(str, 0, track);
        cout << res[pos] << endl;
    }
}
