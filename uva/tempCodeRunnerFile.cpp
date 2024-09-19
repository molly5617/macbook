#include <iostream>
using namespace std;
#include <vector>
#include <map>
#include <set>
int main()
{
    int times;
    cin >> times;
    while (times--)
    {
        int n, m, l;
        cin >> n >> m >> l;
        vector<vector<int>> v;
        map<int, vector<int>> mp;
        set<int> down;
        v.clear();
        mp.clear();
        down.clear();
        while (m--)
        {
            vector<int> temp;
            int a, b;
            cin >> a >> b;
            mp[a].push_back(b);
        }
        vector<int> hand;
        while (l--)
        {
            int temp;
            cin >> temp;
            hand.push_back(temp);
        }
        for (auto i : hand)
        {
            down.insert(i);
            for (int m : mp[i])
            {
                down.insert(m);
                for (auto j : mp[m])
                {
                    down.insert(j);
                }
            }
        }
        cout << down.size() << endl;
    }
}