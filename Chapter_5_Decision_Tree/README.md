**Decision Tree**

Vẫn là bài toán phân loại thư rác đã được nhắc đến ở bài 1, nhưng ở một mức độ cao hơn, một bức thư là lừa đảo nếu nó có chứa từ "Nigerian" và "prince"

Decision Tree là một cây điều kiện, với các nút là điều kiện và con của nó sẽ là 2 nhánh "true" hoặc "false" tương ứng với việc điều kiện có được đáp ứng hay không.

Ví dụ với việc kiểm tra thư rác, chúng ta sẽ có điều kiện là bức thư có chứa từ "nigerian prince" không bằng cách đếm số từ "nigerian" và "prince". Nếu không có thì chúng ta sẽ gán nhãn "No" cho bức thư, ngược lại chúng ta sẽ kiểm tra đến khi bức thư an toàn.

Việc khó khăn của decision tree ở chỗ lấy ra thuộc tính phù hợp.

Xét mẫu thư lừa đảo sau:


```python
data = [
'I am Mohammed Abacha, the son of the late Nigerian Head of '
'State who died on the 8th of June 1998. Since i have been '
'unsuccessful in locating the relatives for over 2 years now '
'I seek your consent to present you as the next of kin so '
'that the proceeds of this account valued at US$15.5 Million '
'Dollars can be paid to you. If you are capable and willing '
'to assist, contact me at once via email with following '
'details: 1. Your full name, address, and telephone number. '
'2. Your Bank Name, Address. 3.Your Bank Account Number and '
'Beneficiary Name - You must be the signatory.'
]

```

Vector hóa email:


```python
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(data)

vec.get_feature_names()[:5]
```

    /usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)





    ['15', '1998', '8th', 'abacha', 'account']



Từ đó chúng ta có thể tìm kiếm từ "nigerian" và "prince"


```python
'nigerian' in vec.get_feature_names()
```

    /usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)





    True




```python
'prince' in vec.get_feature_names()
```

    /usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)





    False



Chúng ta có thể thấy từ "prince" không được tìm thấy trong email, tuy nhiên đây vẫn là thư rác vì có chứa từ "head of state", nó cũng tương đương với từ "prince"

Đây là một vấn đề của feature engineering, yêu cầu người dùng phải trích xuất thuộc tính chuẩn xác. May mắn thay, khung lý thuyết đằng sau decision tree cung cấp một số thứ có thể giải quyết vấn đề này

Để hiểu sâu về decision tree, chúng ta đi vào ví dụ cụ thể hơn:


# Dự đoán loại thuốc cho bệnh nhân

Thông tin về code nằm ở file đính kèm
