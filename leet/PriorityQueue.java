import java.util.TreeSet;

class Solution {
    public int[] solution(String[] operations) throws Exception {
        TreeSet<Integer> res = new TreeSet<>();

        for (String oper : operations) {
            String type = oper.split(" ")[0];
            Integer num = Integer.valueOf(oper.split(" ")[1]);

            switch (type) {
                case "I":
                    res.add(num);
                    break;
                case "D":
                    if (num == -1) { //최솟값 삭제
                        removeMin(res);
                    } else if (num == 1) {
                        removeMax(res);
                    } else {
                        throw new Exception("err num" + num);    
                    }
                    break;
                default:
                    throw new Exception("err");
            }
        }
        int[] answer;
        if (res.isEmpty()) {
            answer = new int[] {0, 0};    
        } else {
            answer = new int[] {res.last(), res.first()};    
        }

        return answer; // 최댓값, 최소값
    }

    void removeMin(TreeSet<Integer> res) {
        if (!res.isEmpty()) {
            res.pollFirst(); // 최소값 제거
        }
    }

    void removeMax(TreeSet<Integer> res) {
        if (!res.isEmpty()) {
            res.pollLast();  // 최대값 제거
        }
    }

}
