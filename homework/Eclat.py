from collections import defaultdict
import ast
import time

def get_data(data_set, qid):
    D = []
    data_url = '../data/one-hot/{0}_class_{1}.txt'.format(data_set, qid)
    with open(data_url, 'rt', encoding='utf-8') as fin:
        row, _ = ast.literal_eval(fin.readline())
        for indexs_str in fin:
            indexs = ast.literal_eval(indexs_str)
            D.append(indexs)
    return D[:int(row / 2)]


# (l1, l2) -> l1 交 l2
def inter(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    s1.intersection_update(s2)
    return list(s1)


# 判断两个list前面k项相等
def k_equal(l1, l2, k):
    for i in range(k):
        if l1[i] != l2[i]:
            return False
    return True


class Eclat(object):
    def __init__(self, min_sup, min_conf):
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.rows= None
        self.I = None
        self.FR = list()


    # 2 tid-list
    def vertical(self, D):
        self.rows = len(D)
        tids_list = defaultdict(list)
        I = set()
        for t in D:
            I.update(t)
        self.I = I
        for item in I:
            for tid in range(len(D)):
                t = D[tid]
                if item in t:
                    tids_list[str(item)].append(tid)

        one_itemsets = list()

        for k, v in tids_list.items():
            if len(v) >= self.min_sup:
                one_itemsets.append(int(k))
                self.FR.append(([int(k)], len(v)))

        two_itemsets = list()
        for item_i in one_itemsets:
            for item_j in one_itemsets:
                if item_i < item_j:
                    inter_list = inter(tids_list[str(item_i)], tids_list[str(item_j)])
                    if len(inter_list) >= self.min_sup:
                        two_itemsets.append(([item_i, item_j], inter_list))
                        self.FR.append(([item_i, item_j], len(inter_list)))
        return two_itemsets

    def bottom_up2(self, S):
        T = []
        for ai in S:
            for aj in S:
                if (ai[0][:-1] == aj[0][:-1]) and (aj[0][-1] > ai[0][-1]):
                    LR = inter(ai[1], aj[1])
                    if len(LR) >= self.min_sup:
                        R = ai[0].copy()
                        R.append(aj[0][-1])
                        T.append((R, LR))
                        self.FR.append((R, len(LR)))
        if len(T) > 1:
            self.bottom_up2(T)

    # 3(A, sup, conf)
    def bottom_up(self, S):
        k_puls_1_itemset = []

        k = len(S[0][0])
        start_index = 0
        while start_index < len(S):
            """start_index - end_index 的itemset前k-1项相同"""
            # 定位end_index
            end_index = start_index
            while end_index < len(S):
                if k_equal(S[start_index][0], S[end_index][0], k-1):
                    end_index += 1
                else:
                    break
            end_index -= 1

            # 生成候选集
            for i in range(start_index, end_index):
                for j in range(i+1, end_index+1):
                    inter_list = inter(S[i][1], S[j][1])
                    if len(inter_list) >= self.min_sup:
                        itemset = S[i][0].copy()
                        itemset.append(S[j][0][-1])
                        k_puls_1_itemset.append((itemset, inter_list))
                        self.FR.append((itemset, len(inter_list)))
            start_index = end_index + 1
        if len(k_puls_1_itemset) > 1:
            self.bottom_up(k_puls_1_itemset)

    def eclat(self, D):
        two_itemsets = self.vertical(D)
        start = time.time()
        self.bottom_up(two_itemsets)
        print('cost time: {0}s'.format(time.time()-start))
        return self.FR


def test():
    D = [[1,3,4],
         [2,3,5],
         [1,2,3,5],
         [2,5]]
    D = get_data('train', 201)
    eclat = Eclat(2, 1)
    eclat.eclat(D)
    for freq_itemset in eclat.FR:
        print(freq_itemset)


if __name__ == '__main__':
    test()

