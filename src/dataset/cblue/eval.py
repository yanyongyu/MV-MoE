# ruff: noqa
import copy
import json
from pathlib import Path
import re
import time
import traceback

from rouge_chinese import Rouge
from sklearn.metrics import classification_report
from transformers import BasicTokenizer

basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=True)


# 将符合诊疗决策树约束的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
def nodematrix(tree):
    nodelist = []
    for i in range(len(tree)):
        nodelist.append(tree[i]["role"])
    node_matrix = [[0 for i in range(len(nodelist))] for j in range(len(nodelist))]

    # if len(tree) == 0:
    #    return (node_matrix)

    count = 0
    while nodelist[0] != "D":
        for i in range(len(nodelist)):
            if nodelist[i] == "C":
                flag, leaf1, leaf2 = 0, 0, 0
                for j in range(i + 1, len(nodelist)):
                    if nodelist[j] == "D" and flag == 0:
                        flag = 1
                        leaf1 = j
                    elif nodelist[j] == "X":
                        continue
                    elif nodelist[j] == "D" and flag == 1:
                        # print(i)
                        leaf2 = j
                        nodelist[i] = "D"
                        node_matrix[leaf1][i] = "F"
                        node_matrix[leaf2][i] = "F"
                        node_matrix[i][leaf1] = "L"
                        node_matrix[i][leaf2] = "R"
                        for k in range(i + 1, leaf2 + 1):
                            nodelist[k] = "X"
                        flag = 2
                        break
                    elif nodelist[j] == "C":
                        break
                if flag == 2:
                    break

        count += 1
        if count > 100:
            break

    return node_matrix


# 计算两个节点的距离
def node_dis(node1, node2):
    if node2 is None:
        # node2 = {"role": "", "triples": [], "logical_rel": ""}
        node2 = {"role": "", "triples": [], "logical_rel": "null"}
    dis = 0
    if node1["role"] != node2["role"]:
        dis += 1
    # print(dis)
    if node1["logical_rel"] != node2["logical_rel"]:
        dis += 1
    dis += len(
        list(
            (set(node1["triples"]) | set(node2["triples"]))
            - (set(node1["triples"]) & set(node2["triples"]))
        )
    )
    return dis


# 判断两条路径是否相同
def is_path_equal(path1, path2):
    if len(path1) != len(path2):
        return False
    for i in range(len(path1)):
        if isinstance(path1[i], dict) and isinstance(path2[i], dict):
            if (
                path1[i]["role"] == path2[i]["role"]
                and path1[i]["logical_rel"] == path2[i]["logical_rel"]
                and set(path1[i]["triples"]) == set(path2[i]["triples"])
            ):
                continue
            else:
                return False
        elif path1[i] != path2[i]:
            return False
    return True


# 判断两棵树是否相同
def is_tree_equal(predict_tree, gold_tree):
    if len(predict_tree) != len(gold_tree):
        return 0
    else:
        for i in range(len(predict_tree)):
            if (
                predict_tree[i]["role"] == gold_tree[i]["role"]
                and predict_tree[i]["logical_rel"] == gold_tree[i]["logical_rel"]
                and set(predict_tree[i]["triples"]) == set(gold_tree[i]["triples"])
            ):
                continue
            else:
                return 0
    return 1


# 计算模型预测的诊疗决策树和ground turth的距离，距离越小表示两树越相似，为计算编辑比率做准备
def edit_distance(predict_tree, gold_tree, predict_matrix, gold_matrix):
    dis = 0
    stack1 = [0]
    stack2 = [0]

    try:
        while stack1:
            s1 = stack1.pop()
            s2 = stack2.pop()
            if ("L" not in predict_matrix[s1] and "R" not in predict_matrix[s1]) and (
                "L" in gold_matrix[s2] or "R" in gold_matrix[s2]
            ):
                dis += node_dis(predict_tree[s1], gold_tree[s2])
                stack_tmp = []
                stack_tmp.append(gold_matrix[s2].index("R"))
                stack_tmp.append(gold_matrix[s2].index("L"))
                while stack_tmp:
                    s_tmp = stack_tmp.pop()
                    dis += node_dis(gold_tree[s_tmp], None)
                    if "L" in gold_matrix[s_tmp] and "R" in gold_matrix[s_tmp]:
                        stack_tmp.append(gold_matrix[s_tmp].index("R"))
                        stack_tmp.append(gold_matrix[s_tmp].index("L"))
            elif ("L" in predict_matrix[s1] and "R" in predict_matrix[s1]) and (
                "L" not in gold_matrix[s2] or "R" not in gold_matrix[s2]
            ):
                dis += node_dis(predict_tree[s1], gold_tree[s2])
                stack_tmp = []
                stack_tmp.append(predict_matrix[s1].index("R"))
                stack_tmp.append(predict_matrix[s1].index("L"))
                while stack_tmp:
                    s_tmp = stack_tmp.pop()
                    dis += node_dis(predict_tree[s_tmp], None)
                    if "L" in predict_matrix[s_tmp] and "R" in predict_matrix[s_tmp]:
                        stack_tmp.append(predict_matrix[s_tmp].index("R"))
                        stack_tmp.append(predict_matrix[s_tmp].index("L"))
            elif ("L" not in predict_matrix[s1] and "R" not in predict_matrix[s1]) and (
                "L" not in gold_matrix[s2] and "R" not in gold_matrix[s2]
            ):
                dis += node_dis(predict_tree[s1], gold_tree[s2])
            else:
                stack1.append(predict_matrix[s1].index("R"))
                stack1.append(predict_matrix[s1].index("L"))
                stack2.append(gold_matrix[s2].index("R"))
                stack2.append(gold_matrix[s2].index("L"))
                dis += node_dis(predict_tree[s1], gold_tree[s2])

    except Exception:
        traceback.print_exc()

    return dis


# 计算决策路径抽取的TP,TP+FP,TP+FN
def decision_path(predict_tree, gold_tree, predict_matrix, gold_matrix):
    leaf1, leaf2, paths1, paths2 = [], [], [], []

    try:
        for i in range(len(predict_matrix)):
            if "L" not in predict_matrix[i] and "R" not in predict_matrix[i]:
                leaf1.append(i)
        for node in leaf1:
            path = [predict_tree[node]]
            while node != 0:
                # print(predict_matrix)
                # print(node)
                # print(predict_matrix[node])
                path.append(predict_matrix[predict_matrix[node].index("F")][node])
                path.append(predict_tree[predict_matrix[node].index("F")])
                node = predict_matrix[node].index("F")
            paths1.append(path)
        for i in range(len(gold_matrix)):
            if "L" not in gold_matrix[i] and "R" not in gold_matrix[i]:
                leaf2.append(i)
        for node in leaf2:
            path = [gold_tree[node]]
            while node != 0:
                path.append(gold_matrix[gold_matrix[node].index("F")][node])
                path.append(gold_tree[gold_matrix[node].index("F")])
                node = gold_matrix[node].index("F")
            paths2.append(path)
        res = 0
        for path1 in paths1:
            for path2 in paths2:
                if is_path_equal(path1, path2):
                    res += 1
                    break
    except Exception:
        traceback.print_exc()
        res = 0

    return res, len(paths1), len(paths2)


# 计算三元组抽取的TP,TP+FP,TP+FN
def triplet_extraction(predict_tree, gold_tree):
    predict_triplet, gold_triplet = [], []
    for i in range(len(predict_tree)):
        for triplet in predict_tree[i]["triples"]:
            predict_triplet.append(triplet)
    for i in range(len(gold_tree)):
        for triplet in gold_tree[i]["triples"]:
            gold_triplet.append(triplet)
    predict_triplet_num = len(list(set(predict_triplet)))
    gold_triplet_num = len(list(set(gold_triplet)))
    correct_triplet_num = len(list(set(gold_triplet) & set(predict_triplet)))
    return [correct_triplet_num, predict_triplet_num, gold_triplet_num]


# 计算节点抽取的TP,TP+FP,TP+FN
def node_extraction(predict_tree, gold_tree):
    predict_node, gold_node = [], []
    for i in range(len(predict_tree)):
        if len(predict_tree[i]["triples"]) > 0:
            predict_node.append(predict_tree[i])
    for i in range(len(gold_tree)):
        if len(gold_tree[i]["triples"]) > 0:
            gold_node.append(gold_tree[i])

    predict_triplet_num = len(predict_node)
    gold_triplet_num = len(gold_node)
    correct_triplet_num = 0
    for node1 in predict_node:
        for node2 in gold_node:
            if (
                len(node1["triples"]) > 0
                and node1["role"] == node2["role"]
                and node1["logical_rel"] == node2["logical_rel"]
                and set(node1["triples"]) == set(node2["triples"])
            ):
                correct_triplet_num += 1
    return [correct_triplet_num, predict_triplet_num, gold_triplet_num]


# 评测函数，共计算5个指标: 三元组抽取的F1；节点抽取的F1；决策树的Acc；决策路径的F1; 树的编辑距离
def text2dt_eval_single_tree(predict_tree, gold_tree):
    # 将符合诊疗决策树的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
    for node in predict_tree:
        for i in range(len(node["triples"])):
            # print(node["triples"][i])
            assert len(node["triples"][i]) == 3, "the triple format is wrong"
            node["triples"][i] = (
                node["triples"][i][0].lower(),
                node["triples"][i][1].lower(),
                node["triples"][i][2].lower(),
            )
    for node in gold_tree:
        for i in range(len(node["triples"])):
            assert len(node["triples"][i]) == 3, "the triple format is wrong"
            node["triples"][i] = (
                node["triples"][i][0].lower(),
                node["triples"][i][1].lower(),
                node["triples"][i][2].lower(),
            )

    # print("step1: ")
    predict_matrix = nodematrix(predict_tree)
    gold_matrix = nodematrix(gold_tree)

    # 用于计算生成树的Acc
    tree_num = 0 if predict_tree == [] else 1
    correct_tree_num = is_tree_equal(predict_tree, gold_tree)

    # 用于计算triplet抽取的F1
    correct_triplet_num, predict_triplet_num, gold_triplet_num = triplet_extraction(
        predict_tree, gold_tree
    )

    # 用于计算决策路径的F1
    # print("step2: ")
    correct_path_num, predict_path_num, gold_path_num = decision_path(
        copy.deepcopy(predict_tree),
        copy.deepcopy(gold_tree),
        copy.deepcopy(predict_matrix),
        copy.deepcopy(gold_matrix),
    )
    # print("correct_path_num: ", correct_path_num)

    # 用于计算树的编辑距离
    edit_dis = edit_distance(predict_tree, gold_tree, predict_matrix, gold_matrix)

    correct_node_num, predict_node_num, gold_node_num = node_extraction(
        predict_tree, gold_tree
    )

    return (
        tree_num,
        correct_tree_num,
        correct_triplet_num,
        predict_triplet_num,
        gold_triplet_num,
        correct_path_num,
        predict_path_num,
        gold_path_num,
        edit_dis,
        correct_node_num,
        predict_node_num,
        gold_node_num,
    )


def calc_info_extract_task_scores(list_structured_golden, list_structured_predict):
    assert len(list_structured_golden) == len(list_structured_predict)

    tp = 0
    fp = 0
    fn = 0
    for samp_golden, samp_predict in zip(
        list_structured_golden, list_structured_predict
    ):
        assert samp_golden["sample_id"] == samp_predict["sample_id"], (
            "sample ordering is wrong!"
        )
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert isinstance(answer_golden, list)
        assert isinstance(answer_predict, list), "sample format is wrong!"

        set_golden = set()
        for inst in answer_golden:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            inst = tuple([json.dumps(inst[w], ensure_ascii=False) for w in keys])
            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_golden.add(inst)

        set_predict = set()
        for inst in answer_predict:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            # inst = tuple([inst[w] for w in keys])
            inst = tuple([json.dumps(inst[w], ensure_ascii=False) for w in keys])

            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_predict.add(inst)

        # print("set_predict: ", set_predict)
        # print("set_golden: ", set_golden)

        tp += len(set_golden.intersection(set_predict))
        fp += len(set_predict.difference(set_golden))
        fn += len(set_golden.difference(set_predict))

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    return precision, recall, f1


def calc_cls_task_scores(
    list_structured_golden,
    list_structured_predict,
    list_labels=None,
    return_macro=False,
):
    # types = list_labels
    # scores = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in list_labels + ["ALL"]}

    predictions = []
    ground_truths = []

    # Count GT relations and Predicted relations
    assert len(list_structured_golden) == len(list_structured_predict)
    n_sents = len(list_structured_golden)

    # Count TP, FP and FN per type
    for pred_samp, gt_samp in zip(list_structured_predict, list_structured_golden):
        assert pred_samp["sample_id"] == gt_samp["sample_id"], (
            "sample ordering is wrong!"
        )

        pred_label = pred_samp["answer"]
        gt_label = gt_samp["answer"]
        assert gt_label != ""
        if pred_label == "":
            pred_label = list_labels[0]

        predictions.append(pred_label)
        ground_truths.append(gt_label)

    # metric
    t0 = time.time()
    cls_report = classification_report(
        ground_truths,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    # print(cls_report)

    t1 = time.time()
    # print("calculation metrics: ", t1 - t0)

    if return_macro:
        return (
            cls_report["macro avg"]["precision"],
            cls_report["macro avg"]["recall"],
            cls_report["macro avg"]["f1-score"],
        )
    else:
        return (
            cls_report["weighted avg"]["precision"],
            cls_report["weighted avg"]["recall"],
            cls_report["weighted avg"]["f1-score"],
        )


def calc_nlg_task_scores(list_structured_golden, list_structured_predict):
    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    for samp_golden, samp_predict in zip(
        list_structured_golden, list_structured_predict
    ):
        # print("samp_golden: ", samp_golden)
        # print("samp_predict: ", samp_predict)

        assert samp_golden["sample_id"] == samp_predict["sample_id"], (
            "sample ordering is wrong!"
        )
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert isinstance(answer_golden, str)
        assert isinstance(answer_predict, str), "sample format is wrong!"

        # basic tokenizer: 拆分中文字，保留英文单词
        answer_predict = basic_tokenizer.tokenize(answer_predict)
        answer_golden = basic_tokenizer.tokenize(answer_golden)
        answer_predict = " ".join(answer_predict).strip()
        answer_golden = " ".join(answer_golden).strip()
        if answer_golden.strip() == "":
            answer_golden = "无 。"
        if answer_predict.strip() == "":
            answer_predict = "无 。"
        # print("answer_predict: ", answer_predict)
        # print("answer_golden: ", answer_golden)

        predictions.append(answer_predict)
        references.append(answer_golden)

    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    return rouge1, rouge2, rougeL


def calc_nlg_task_scores_by_sessions(list_structured_golden, list_structured_predict):
    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    for samp_golden, samp_predict in zip(
        list_structured_golden, list_structured_predict
    ):
        # print("samp_golden: ", samp_golden)
        # print("samp_predict: ", samp_predict)

        assert samp_golden["sample_id"] == samp_predict["sample_id"], (
            "sample ordering is wrong!"
        )
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        # if set(answer_golden.keys()) != set(answer_predict.keys())

        for key in answer_golden.keys():
            pred = answer_predict.get(key, "").strip()
            gt = answer_golden[key].strip()

            # basic tokenizer: 拆分中文字，保留英文单词
            pred = basic_tokenizer.tokenize(pred)
            gt = basic_tokenizer.tokenize(gt)
            pred = " ".join(pred).strip()
            gt = " ".join(gt).strip()
            if gt.strip() == "":
                gt = "无 。"
            if pred.strip() == "":
                pred = "无 。"

            # if gt != pred:
            #     print(gt)
            #     print(pred)

            predictions.append(pred)
            references.append(gt)

    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    return rouge1, rouge2, rougeL


def calc_text2dt_task_scores(
    list_structured_golden,
    list_structured_predict,
):
    assert len(list_structured_golden) == len(list_structured_predict)

    gold_tree_num, correct_tree_num = 0.000001, 0.000001
    gold_triplet_num, predict_triplet_num, correct_triplet_num = (
        0.000001,
        0.000001,
        0.000001,
    )
    gold_path_num, predict_path_num, correct_path_num = 0.000001, 0.000001, 0.000001
    gold_node_num, predict_node_num, correct_node_num = 0.000001, 0.000001, 0.000001

    edit_dis = 0
    max_edit_dis = 0

    for samp_golden, samp_predict in zip(
        list_structured_golden, list_structured_predict
    ):
        assert samp_golden["sample_id"] == samp_predict["sample_id"], (
            "sample ordering is wrong!"
        )
        tree_golden = samp_golden["answer"]
        tree_predict = samp_predict["answer"]

        assert isinstance(tree_golden, list)
        assert isinstance(tree_predict, list), "sample format is wrong!"

        tmp = text2dt_eval_single_tree(tree_predict, tree_golden)
        gold_tree_num += tmp[0]
        correct_tree_num += tmp[1]
        correct_triplet_num += tmp[2]
        predict_triplet_num += tmp[3]
        gold_triplet_num += tmp[4]
        correct_path_num += tmp[5]
        predict_path_num += tmp[6]
        gold_path_num += tmp[7]
        edit_dis += tmp[8]

        # 计算最大编辑数
        max_edit_dis += (tmp[3] + tmp[10] * 2) + (tmp[4] + tmp[11] * 2)

        correct_node_num += tmp[9]
        predict_node_num += tmp[10]
        gold_node_num += tmp[11]

    tree_acc = correct_tree_num / gold_tree_num
    triplet_f1 = (
        2
        * (correct_triplet_num / predict_triplet_num)
        * (correct_triplet_num / gold_triplet_num)
        / (
            correct_triplet_num / predict_triplet_num
            + correct_triplet_num / gold_triplet_num
        )
    )
    path_f1 = (
        2
        * (correct_path_num / predict_path_num)
        * (correct_path_num / gold_path_num)
        / (correct_path_num / predict_path_num + correct_path_num / gold_path_num)
    )
    tree_lenv_radio = 1 - edit_dis / max_edit_dis
    node_f1 = (
        2
        * (correct_node_num / predict_node_num)
        * (correct_node_num / gold_node_num)
        / (correct_node_num / predict_node_num + correct_node_num / gold_node_num)
    )

    return tree_lenv_radio, node_f1, path_f1


# 错误字典，这里只是示例
error_msg = {
    1: "There are missing predictions in the submission, please check again!",
    2: "Predictions are in the wrong format, please check again! ",
    3: "It seems there are missing predictions or the predicted samples are in the wrong order, please check again! ",
    4: "Error in calculating metrics!",
    10: "test_predictions.json file not in submission, please check again!",
    11: "results.json file not in submission, please check again!",
    12: "post_generate_process.py file not in submission, please check again!",
    13: "loading results.json file fails, please check again!",
    99: "Other error unknown!",
}


# def dump_2_json(info, path):
#     with open(path, "w") as output_json_file:
#         json.dump(info, output_json_file, indent=2, ensure_ascii=False)


# def report_error_msg(detail, show_msg, out_p):
#     error_dict = dict()
#     error_dict["errorDetail"] = detail
#     error_dict["errorMsg"] = show_msg
#     error_dict["score"] = 0
#     error_dict["scoreJson"] = {}
#     error_dict["success"] = False
#     dump_2_json(error_dict, out_p)


# def report_score(score_map, out_p):
#     result = dict()
#     result["success"] = True

#     result["score"] = score_map["score"]
#     result["scoreJson"] = score_map

#     # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
#     # result['scoreJson'] = {'score': score, 'aaaa': 0.1}
#     # result['scoreJson'] = {'score': score}

#     dump_2_json(result, out_p)


def calc_scores(dict_gt, dict_pred):
    scores = {
        "CMeEE-V2": {},
        "CMeIE-V2": {},
        "CHIP-CDN": {},
        "CHIP-CDEE": {},
        "IMCS-V2-NER": {},
        "CHIP-MDCFNPC": {},
        "IMCS-V2-SR": {},
        "IMCS-V2-DAC": {},
        "IMCS-V2-MRG": {},
        "CHIP-CTC": {},
        "CHIP-STS": {},
        "KUAKE-IR": {},
        "KUAKE-QIC": {},
        "KUAKE-QQR": {},
        "KUAKE-QTR": {},
        "MedDG": {},
        "Text2DT": {},
        "CMedCausal": {},
    }

    for task_name in scores.keys():
        assert task_name in dict_gt

        if task_name not in dict_pred or not dict_pred[task_name]:
            continue

        gts = dict_gt[task_name]
        preds = dict_pred[task_name]
        if not len(gts) == len(preds):
            raise ValueError(error_msg[1])

        for gt_inst, pred_inst in zip(gts, preds):
            if not isinstance(pred_inst, dict):
                raise ValueError(error_msg[2])

            if "sample_id" not in pred_inst:
                raise ValueError(error_msg[2])

            if gt_inst.get("sample_id") != pred_inst.get("sample_id"):
                raise ValueError(error_msg[3])

        if task_name in [
            "CMeEE-V2",
            "CMeIE-V2",
            "CHIP-CDN",
            "CMedCausal",
            "CHIP-CDEE",
            "IMCS-V2-NER",
            "IMCS-V2-SR",
            "CHIP-MDCFNPC",
        ]:
            try:
                precision, recall, f1 = calc_info_extract_task_scores(gts, preds)
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            except Exception as e:
                raise ValueError(error_msg[4]) from e

        # "CHIP-STS"
        elif task_name in [
            "CHIP-STS",
        ]:
            try:
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=["是的", "不是"],
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "CHIP-CTC",
        ]:
            try:
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=[
                        "非上述类型",
                        "疾病",
                        "症状(患者感受)",
                        "体征(医生检测）",
                        "怀孕相关",
                        "肿瘤进展",
                        "疾病分期",
                        "过敏耐受",
                        "器官组织状态",
                        "预期寿命",
                        "口腔相关",
                        "药物",
                        "治疗或手术",
                        "设备",
                        "护理",
                        "诊断",
                        "实验室检查",
                        "风险评估",
                        "受体状态",
                        "年龄",
                        "特殊病人特征",
                        "读写能力",
                        "性别",
                        "教育情况",
                        "居住情况",
                        "种族",
                        "知情同意",
                        "参与其它试验",
                        "研究者决定",
                        "能力",
                        "伦理审查",
                        "依存性",
                        "成瘾行为",
                        "睡眠",
                        "锻炼",
                        "饮食",
                        "酒精使用",
                        "性取向",
                        "吸烟状况",
                        "献血",
                        "病例来源",
                        "残疾群体",
                        "健康群体",
                        "数据可及性",
                        "含有多个类别",
                    ],
                    return_macro=True,
                )
                scores[task_name] = {
                    "macro-precision": precision,
                    "macro-recall": recall,
                    "macro-f1": f1,
                }

            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "IMCS-V2-DAC",
        ]:
            # TODO: 查看样本不均衡性
            try:
                list_labels = [
                    "非上述类型",
                    "关于症状的询问",
                    "关于症状的回答",
                    "关于病因的询问",
                    "关于病因的回答",
                    "关于个人基本信息的询问",
                    "关于个人基本信息的回答",
                    "关于已有检查和治疗的提问",
                    "关于已有检查和治疗的回答",
                    "关于用药建议的提问",
                    "关于用药建议的解答",
                    "关于就医建议的提问",
                    "关于就医建议的解答",
                    "关于注意事项的提问",
                    "关于注意事项的解答",
                    "给出诊断",
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=True,
                )
                scores[task_name] = {
                    "macro-precision": precision,
                    "macro-recall": recall,
                    "macro-f1": f1,
                }

            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "KUAKE-IR",
        ]:
            try:
                list_labels = ["相关", "不相关"]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "KUAKE-QIC",
        ]:
            try:
                list_labels = [
                    "非上述类型",
                    "病情诊断",
                    "病因分析",
                    "治疗方案",
                    "就医建议",
                    "指标解读",
                    "疾病描述",
                    "后果表述",
                    "注意事项",
                    "功效作用",
                    "医疗费用",
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=True,
                )
                scores[task_name] = {
                    "macro-precision": precision,
                    "macro-recall": recall,
                    "macro-f1": f1,
                }

            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "KUAKE-QTR",
        ]:
            try:
                list_labels = [
                    "完全不匹配或者没有参考价值",
                    "很少匹配有一些参考价值",
                    "部分匹配",
                    "完全匹配",
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "KUAKE-QQR",
        ]:
            try:
                list_labels = [
                    "完全一致",
                    "后者是前者的语义子集",
                    "后者是前者的语义父集",
                    "语义无直接关联",
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "MedDG",
        ]:
            try:
                rouge1, rouge2, rougeL = calc_nlg_task_scores(
                    gts,
                    preds,
                )
                # print("rouge1: ", rouge1)
                # print("rouge2: ", rouge2)
                # print("rougeL: ", rougeL)
                scores[task_name] = {
                    "rouge1": rouge1,
                    "rouge2": rouge2,
                    "rougeL": rougeL,
                }
            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "IMCS-V2-MRG",
        ]:
            try:
                rouge1, rouge2, rougeL = calc_nlg_task_scores_by_sessions(
                    gts,
                    preds,
                )
                scores[task_name] = {
                    "rouge1": rouge1,
                    "rouge2": rouge2,
                    "rougeL": rougeL,
                }
            except Exception as e:
                raise ValueError(error_msg[4]) from e

        elif task_name in [
            "Text2DT",
        ]:
            try:
                tree_lenv_radio, node_f1, path_f1 = calc_text2dt_task_scores(
                    gts,
                    preds,
                )
                scores[task_name] = {
                    "tree_lenv_radio": tree_lenv_radio,
                    "node_f1": node_f1,
                    "path_f1": path_f1,
                }
            except Exception as e:
                raise ValueError(error_msg[4]) from e

    # 计算average score
    total_score = 0.0
    for task_name in scores.keys():
        if "rougeL" in scores[task_name]:
            total_score += scores[task_name].get("rougeL", 0.0)
        elif "macro-f1" in scores[task_name]:
            total_score += scores[task_name].get("macro-f1", 0.0)
        elif "tree_lenv_radio" in scores[task_name]:
            total_score += scores[task_name].get("tree_lenv_radio", 0.0)
        else:
            total_score += scores[task_name].get("f1", 0.0)

    scores["Overall"] = total_score / len(scores.keys())
    print("scores for all tasks: ", scores)  # noqa: T201

    # 修改score的格式
    score_map = {}
    for task_name in scores.keys():
        if task_name == "Overall":
            continue

        if task_name in ["CHIP-CTC", "KUAKE-QIC", "IMCS-V2-DAC"]:
            score_map[f"{task_name}-Macro-F1"] = scores[task_name].get("macro-f1", 0.0)
        elif task_name in ["MedDG", "IMCS-V2-MRG"]:
            score_map[f"{task_name}-RougeL"] = scores[task_name].get("rougeL", 0.0)
        elif task_name in [
            "Text2DT",
        ]:
            score_map[f"{task_name}-TreeLenvRatio"] = scores[task_name].get(
                "tree_lenv_radio", 0.0
            )
        else:
            score_map[f"{task_name}-Micro-F1"] = scores[task_name].get("f1", 0.0)
    score_map["score"] = scores["Overall"]

    return {key: value * 100 for key, value in score_map.items()}


def process_cmeee_v2(output: str, answer_choices: list[str]) -> list[dict]:
    # 答案格式：
    #   第一行：引导词
    #   实体每类占一行，每行格式为 "[类型名称]实体：实体名称1，实体名称2，实体名称3\n"
    #                多个实体，用 ， 符号分割

    list_entities: list[dict] = []
    for choice in answer_choices:
        for piece in output.split("\n"):
            match = re.search(rf"{choice}实体[:：](.+)", piece)
            if not match:
                continue
            mentions = re.split(r"[,，、；；]", match[1].strip())
            mentions = [w.strip() for w in mentions if len(w.strip()) > 0]
            for ment in mentions:
                list_entities.append({"entity": ment, "type": choice})
    return list_entities


def process_cmeie_v2(output: str) -> list[dict]:
    # 答案格式：
    #   每个关系类型占一行，格式为
    #         "具有{lab}关系的实体对如下：头实体：str，尾实体：str；头实体：str，尾实体：str；"  # noqa: E501

    list_spos: list[dict] = []
    for line in output.split("\n"):
        # print("line: ", line)
        # 首先是解析出label:
        match = re.search(r"具有(\w+?)关系的.+?[:：](.+)", line)
        if not match:
            continue

        predicate = match[1]
        line = match[2]

        for spo_str in re.split(r"[;；.。]", line):
            match = re.search(r"头实体为(.+?)[,，]\s?尾实体为([^.。]+)", spo_str)
            if not match:
                continue

            head_mention_str = match[1].strip()
            tail_mention_str = match[2].strip()

            list_spos.append(
                {
                    "predicate": predicate,
                    "subject": head_mention_str,
                    "object": tail_mention_str,
                }
            )
    return list_spos


def process_cmedcausal(output: str) -> list[dict]:
    list_spos: list[dict] = []
    list_answer_strs = output.split("\n")

    rel = None
    for ans_str in list_answer_strs:
        ans_str = ans_str.strip()
        if not ans_str:
            continue

        if "因果关系" in ans_str:
            rel = "因果关系"
            continue
        elif "条件关系" in ans_str:
            rel = "条件关系"
            continue
        elif "上下位关系" in ans_str:
            rel = "上下位关系"
            continue
        elif "给定的文本没有指定关系的三元组" in ans_str:
            break

        if rel != "条件关系":
            match = re.search(r"头实体[:：](.+?)[,，;；]尾实体[:：]([^.。]+)", ans_str)
            if not match:
                continue
            triple = {
                "predicate": rel,
                "subject": match[1].strip(),
                "object": match[2].strip(),
            }

        else:
            # 头实体：不治疗；尾三元组：头实体：急性痛风关节炎；尾实体：红、肿、热、痛5-7天可以明显好转；关系：因果关系  # noqa: E501
            match = re.search(
                r"头实体[:：](.+?)[,，;；]尾三元组[:：]头实体[:：](.+?)[,，;；]尾实体[:：](.+?)[,，;；]关系[:：]([^.。]+)",
                ans_str,
            )
            if not match:
                continue
            triple = {
                "predicate": rel,
                "subject": match[1].strip(),
                "object": {
                    "predicate": match[4].strip(),
                    "subject": match[2].strip(),
                    "object": match[3].strip(),
                },
            }

        list_spos.append(triple)
    return list_spos


def process_text2dt(output: str) -> list[dict]:
    list_nodes: list[dict] = []

    for ans_str in output.split("\n"):
        if not ans_str:
            continue

        # 节点0：role=C；logical_rel=null；triples=[["AVNRT患者", "临床表现", "低血压"]]
        match = re.search(
            r"role=([^;；]+)[;；]logical_rel=([^;；]+)[;；]triples=(.+)", ans_str
        )
        if not match:
            continue

        role = match[1].strip()
        logical_rel = match[2].strip()
        triples = match[3].strip()

        try:
            triples = eval(triples)
            if not isinstance(triples, list):
                triples = []

            triples_new = []
            for tri in triples:
                if len(tri) != 3:
                    continue
                if not all(isinstance(w, str) for w in tri):
                    continue
                triples_new.append(tri)
        except Exception:
            traceback.print_exc()
            triples_new = []

        node = {
            "role": role,
            "logical_rel": logical_rel,
            "triples": triples_new,
        }
        list_nodes.append(node)

    if not list_nodes:
        list_nodes = [{"role": "D", "logical_rel": "null", "triples": []}]

    return list_nodes


def process_chip_cdn(output: str, answer_choices: list[str]) -> list[dict]:
    # 答案格式：
    #   多个选中的标准化实体，用 ， 符号分割

    answers = []
    for choice in answer_choices:
        if choice in output:
            answers.append(choice)
    answers = list(set(answers))
    answers = [{"entity": w, "type": "normalization"} for w in answers]
    return answers


def process_chip_cdee(output: str) -> list[dict]:
    # 答案格式：
    #   第一行：引导词
    #   每个事件占一行，事件字段用 ； 分隔， 然后每个字段是 字段名：字段值的格式"
    #                                  字段值有多个，则用 ，符号分隔

    # 主体词：癌；发生状态：；描述词：术后放化疗后；解剖部位：乳腺
    keys = ["主体词", "发生状态", "描述词", "解剖部位"]

    events: list[dict] = []
    for ans_str in output.split("\n"):
        event_info = {}
        for attr_str in re.split(r"[;；.。]", ans_str):
            for key in keys:
                match = re.search(rf"{key}[:：](.+)", attr_str)
                if not match:
                    continue
                a_attr = match[1].strip()
                if key in ["描述词", "解剖部位"]:
                    a_attr_split = re.split(r"[,，]", a_attr)
                    a_attr_split = [
                        w.strip() for w in a_attr_split if len(w.strip()) > 0
                    ]
                    event_info[key] = a_attr_split
                else:
                    event_info[key] = a_attr

        for key in keys:
            if key not in event_info:
                if key in ["描述词", "解剖部位"]:
                    event_info[key] = []
                else:
                    event_info[key] = ""

        events.append(event_info)
    return events


def process_chip_sts(output: str) -> str:
    # 答案格式：直接回答"是"，"不是"，"相同"， ”不同“
    answer_str = output.split("\n")[0].strip()
    if "相同" in answer_str or "是的" in answer_str:
        return "是的"
    elif "不同" in answer_str or "不是" in answer_str:
        return "不是"
    return answer_str


def process_chip_ctc(output: str, answer_choices: list[str]) -> str:
    # 答案格式：直接回答分类标签
    for choice in answer_choices:
        if choice in output:
            return choice
    return output.strip()


def process_kuake_ir(output: str) -> str:
    # 答案格式：直接回答 "相关", "不相关"
    answer_str = output.split("\n")[0].strip()
    if "不相关" in answer_str:
        return "不相关"
    elif "相关" in answer_str:
        return "相关"
    return answer_str


def process_kuake_qic(output: str, answer_choices: list[str]) -> str:
    # 答案格式：直接回答分类标签
    for choice in answer_choices:
        if choice in output:
            return choice
    return output.strip()


def process_kuake_qqr(output: str, answer_choices: list[str]) -> str:
    # 答案格式：直接回答分类标签
    for choice in answer_choices:
        if choice in output:
            return choice
    return output.strip()


def process_kuake_qtr(output: str, answer_choices: list[str]) -> str:
    # 答案格式：直接回答分类标签
    for choice in answer_choices:
        if choice in output:
            return choice
    return output.strip()


def process_chip_mdcfnpc(output: str, answer_choices: list[str]) -> list[dict]:
    # 答案格式：
    #   第一行：引导词
    #    每一行就是 "[症状词]：[阴阳性判断结果]"
    list_finding_attrs = []
    for ans_str in output.split("\n"):
        match = re.search(r"(\w+?)[:：]([^.。]+)", ans_str)
        if not match:
            continue
        finding, conclusion = match[1].strip(), match[2].strip()

        if conclusion not in answer_choices:
            conclusion = "不标注"

        list_finding_attrs.append({"entity": finding, "attr": conclusion})
    return list_finding_attrs


def process_imcs_v2_ner(output: str, answer_choices: list[str]) -> list[dict]:
    # 答案格式：
    #   第一行：引导词
    #   实体每类占一行，每行格式为 "[类型名称]实体：实体名称1，实体名称2，实体名称3\n"
    #                多个实体，用 ， 符号分割

    list_entities: list[dict] = []
    assert isinstance(answer_choices, list)
    for choice in answer_choices:
        for piece in output.split("\n"):
            match = re.search(rf"{choice}实体[:：]([^.。]+)", piece)
            if not match:
                continue
            mentions = re.split(r"[,，]", match[1].strip())
            mentions = [w.strip() for w in mentions if len(w.strip()) > 0]
            for ment in mentions:
                list_entities.append(
                    {
                        "entity": ment,
                        "type": choice,
                    }
                )
    return list_entities


def process_imcs_v2_dac(output: str, answer_choices: list[str]) -> str:
    # 答案格式：直接回答分类标签
    for choice in answer_choices:
        if choice in output:
            return choice
    return output.strip()


def process_imcs_v2_sr(output: str, answer_choices: list[str]) -> list[dict]:
    # 答案格式：
    #   第一行：引导词
    #    每一行就是 "[症状词]：[阴阳性判断结果]"
    list_finding_attrs = []
    for ans_str in output.split("\n"):
        match = re.search(r"(\w+?)[:：]([^.。]+)", ans_str)
        if not match:
            continue
        finding, conclusion = match[1].strip(), match[2].strip()

        if conclusion not in answer_choices:
            conclusion = "无法根据上下文确定病人是否患有该症状"

        list_finding_attrs.append({"entity": finding, "attr": conclusion})
    return list_finding_attrs


def process_imcs_v2_mrg(output: str, answer_choices: list[str]) -> dict:
    # 答案格式：
    #   1. 第一行是引导词；
    #   第二行开始是 [section_name]：str的格式

    keys = ["主诉", "现病史", "辅助检查", "既往史", "诊断", "建议"]

    answer_dict = {}
    for line in output.split("\n"):
        # print("line: ", line)
        match = re.search(rf"({'|'.join(keys)})[:：](.+)", line)
        if not match:
            continue
        key = match[1].strip()
        answer_dict[key] = match[2].strip()
    for key in keys:
        if key not in answer_dict:
            answer_dict[key] = ""
    return answer_dict


def process_meddg(output: str) -> str:
    # 答案格式：str
    return output.strip()


def process_generated_results(pred_file: str | list[dict]):
    structured_output = {
        "CMeEE-V2": [],
        "CMeIE-V2": [],
        "CHIP-CDN": [],
        "CHIP-CDEE": [],
        "CHIP-STS": [],
        "CHIP-CTC": [],
        "CHIP-MDCFNPC": [],
        "KUAKE-IR": [],
        "KUAKE-QIC": [],
        "KUAKE-QQR": [],
        "KUAKE-QTR": [],
        "MedDG": [],
        "IMCS-V2-MRG": [],
        "IMCS-V2-NER": [],
        "IMCS-V2-DAC": [],
        "IMCS-V2-SR": [],
        "Text2DT": [],
        "CMedCausal": [],
    }

    if isinstance(pred_file, str):
        data = [json.loads(line) for line in Path(pred_file).read_text().splitlines()]
    else:
        data = pred_file

    for line in data:
        sample_id_ = line.get("sample_id", "xxxx")
        # input = line["input"]
        gen_output = line["target"]
        # gen_output = (
        #     gen_output.replace(":", "：", 100)
        #     .replace(",", "，", 100)
        #     .replace(";", "；", 100)
        # )
        # gen_output = line["generated_output"]
        task_dataset = line["task_dataset"]
        # task_type = line["task_type"]

        # 选项：
        answer_choices = line["answer_choices"]

        if task_dataset == "CMeEE-V2":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_cmeee_v2(gen_output, answer_choices),
                }
            )

        elif task_dataset == "CMeIE-V2":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_cmeie_v2(gen_output),
                }
            )

        elif task_dataset == "CMedCausal":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_cmedcausal(gen_output),
                }
            )

        elif task_dataset == "Text2DT":
            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_text2dt(gen_output),
                }
            )

        elif task_dataset == "CHIP-CDN":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_chip_cdn(gen_output, answer_choices),
                }
            )

        elif task_dataset == "CHIP-CDEE":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_chip_cdee(gen_output),
                }
            )

        elif task_dataset == "CHIP-STS":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_chip_sts(gen_output),
                }
            )

        elif task_dataset == "CHIP-CTC":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_chip_ctc(gen_output, answer_choices),
                }
            )

        elif task_dataset == "KUAKE-IR":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_kuake_ir(gen_output),
                }
            )

        elif task_dataset == "KUAKE-QIC":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_kuake_qic(gen_output, answer_choices),
                }
            )

        elif task_dataset == "KUAKE-QQR":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_kuake_qqr(gen_output, answer_choices),
                }
            )

        elif task_dataset == "KUAKE-QTR":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_kuake_qtr(gen_output, answer_choices),
                }
            )

        elif task_dataset == "CHIP-MDCFNPC":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_chip_mdcfnpc(gen_output, answer_choices),
                }
            )

        elif task_dataset == "IMCS-V2-NER":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_imcs_v2_ner(gen_output, answer_choices),
                }
            )

        elif task_dataset == "IMCS-V2-DAC":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_imcs_v2_dac(gen_output, answer_choices),
                }
            )

        elif task_dataset == "IMCS-V2-SR":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_imcs_v2_sr(gen_output, answer_choices),
                }
            )

        elif task_dataset == "IMCS-V2-MRG":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_imcs_v2_mrg(gen_output, answer_choices),
                }
            )

        elif task_dataset == "MedDG":
            structured_output[task_dataset].append(
                {
                    "sample_id": sample_id_,
                    "answer": process_meddg(gen_output),
                }
            )

        else:
            raise ValueError(f"Unknown task_dataset: {task_dataset}")

    return structured_output
