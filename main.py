import datetime

import google_auth_httplib2
import httplib2
import pandas as pd
import plotly.express as px
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import HttpRequest
from sklearn.linear_model import LinearRegression

import data as d

SCOPE = "https://www.googleapis.com/auth/spreadsheets"
SHEET_ID = "1Sx0MFwfZdgam9cBHzgo97XDqArOMz7czS4BrX7niPIc"
SHEET_NAME = "multi1"
X_COLS = [
    "身長",
    "体重",
    "座高",
    "握力",
    "上体起こし",
    "長座体前屈",
    "反復横跳び",
    "シャトルラン",
    "50ｍ走",
    "立ち幅跳び",
    "ハンドボール投げ",
]
TEST_START_INDEX = 400
TEST_END_INDEX = 420
# 花子のデータ：身長, 体重, 座高, 握力, 上体起こし, 長座体前屈, 反復横跳び, シャトルラン, 50ｍ走, 立ち幅跳び, ハンドボール投げ
target_df = pd.DataFrame(
    [[162.8, 49.7, 87.3, 23, 26, 41, 44, 67, 8.3, 0.0, 14]], columns=X_COLS
)


# @st.experimental_singleton()
# def connect_to_gsheet():
#     # Create a connection object
#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"], scopes=[SCOPE]
#     )

#     # Create a new Http() object for every request
#     def build_request(http, *args, **kwargs):
#         new_http = google_auth_httplib2.AuthorizedHttp(
#             credentials, http=httplib2.Http()
#         )

#         return HttpRequest(new_http, *args, **kwargs)

#     authorized_http = google_auth_httplib2.AuthorizedHttp(
#         credentials, http=httplib2.Http()
#     )

#     service = build("sheets", "v4", requestBuilder=build_request, http=authorized_http)
#     gsheet_connector = service.spreadsheets()

#     return gsheet_connector


# def add_row_to_gsheet(gsheet_connector, row):
#     gsheet_connector.values().append(
#         spreadsheetId=SHEET_ID,
#         range=f"{SHEET_NAME}!A:G",
#         body=dict(values=row),
#         valueInputOption="USER_ENTERED",
#     ).execute()


@st.cache
def load_full_data():
    data = pd.read_csv(d.DATA_SOURCE)
    return data


@st.cache
def load_num_data():
    data = pd.read_csv(d.DATA_SOURCE)
    rows = ["学年", "性別"]
    data = data.drop(rows, axis=1)
    return data


def main():
    # if "page" not in st.session_state:
    #     st.session_state.page = "vis"

    # ログをとるときのみコメントを外す
    # If username is already initialized, don't do anything
    if "username" not in st.session_state or st.session_state.username == "default":
        st.session_state.username = "default"
        input_name()
        st.stop()
    if "username" not in st.session_state:
        st.session_state.username = "test"

    # 個別のログをとるときはinputを受け取るので以下は不要
    # st.session_state.username = 'test'
    # if 'page' not in st.session_state:
    #     st.session_state.page = 'input_name' # usernameつける時こっち

    # --- page選択ラジオボタン
    st.sidebar.markdown("## Select Mode")
    st.session_state.page = st.sidebar.radio("ページ選択", ("Simple Regression", "Data Visualization", "Multiple Regression"))
#     st.session_state.page = st.sidebar.radio("ページ選択", ("Data Visualization", "Simple Regression"))

    # --- page振り分け
    if st.session_state.page == "input_name":
        input_name()
    elif st.session_state.page == "Data Visualization":
        st.session_state.page = "vis"
        vis()
    elif st.session_state.page == "Simple Regression":
        st.session_state.page = "lr"
        lr()
    elif st.session_state.page == "Multiple Regression":
        st.session_state.page = "lr"
        multi_lr()


# ---------------- usernameの登録 ----------------------------------
def input_name():
    # Input username
    with st.form("my_form"):
        inputname = st.text_input("Your Name", placeholder="Input your name here")
        submitted = st.form_submit_button("Go!!")

        # usernameが未入力でないか確認
        if inputname == "":
            submitted = False

        # Goボタンが押されたときの処理
        if submitted:
            st.session_state.username = inputname
            st.session_state.page = "deal_data"
            st.write("Your name: ", inputname)
            st.text("↑ Confirm your name, and press Go! again!")


# ---------------- グラフで可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("PE test Data")
    score = load_num_data()
    full_data = load_full_data()

    st.sidebar.markdown("## Various Visualization Methods")

    # sidebar でグラフを選択
    graph = st.sidebar.radio("Visualization Methods", ("ScatterPlot", "Histogram", "BoxPlot"))

    if graph == "ScatterPlot":
        left, right = st.columns(2)

        with left:  # ScatterPlotの表示
            x_label = st.selectbox("Select y-axis", X_COLS)
            y_label = st.selectbox("Select x-axis", X_COLS)

        with right:  # 色分けオプション
            coloring = st.radio("Coloring", ("None", "Grade", "Sex"))

        if coloring == "Grade":
            fig = px.scatter(full_data, x=x_label, y=y_label, color="学年")
        elif coloring == "Sex":
            fig = px.scatter(
                full_data,
                x=x_label,
                y=y_label,
                color="性別",
            )
        else:
            fig = px.scatter(
                full_data,
                x=x_label,
                y=y_label,
            )

        # 相関係数算出
        cor = d.get_corrcoef(score, x_label, y_label)
        st.write("Correlation Coefficient: " + str(cor))

        # グラフ描画
        st.plotly_chart(fig, use_container_width=True)

        # ログを記録
        # add_row_to_gsheet(
        #     gsheet_connector,
        #     [
        #         [
        #             datetime.datetime.now(
        #                 datetime.timezone(datetime.timedelta(hours=9))
        #             ).strftime("%Y-%m-%d %H:%M:%S"),
        #             st.session_state.username,
        #             "ScatterPlot",
        #             x_label,
        #             y_label,
        #             coloring,
        #         ]
        #     ],
        # )

    # Histogram
    elif graph == "Histogram":
        hist_val = st.selectbox("Select a Variable", X_COLS)
        fig = px.histogram(score, x=hist_val)
        st.plotly_chart(fig, use_container_width=True)

        # ログを記録
        # add_row_to_gsheet(
        #     gsheet_connector,
        #     [
        #         [
        #             datetime.datetime.now(
        #                 datetime.timezone(datetime.timedelta(hours=9))
        #             ).strftime("%Y-%m-%d %H:%M:%S"),
        #             st.session_state.username,
        #             "Histogram",
        #             hist_val,
        #             "-",
        #             "-",
        #         ]
        #     ],
        # )

    # BoxPlot
    elif graph == "BoxPlot":
        box_val_y = st.selectbox("Select a Variable for BoxPlot", X_COLS)

        left, right = st.columns(2)
        with left:  # ScatterPlotの表示
            fig = px.box(
                full_data,
                x="学年",
                y=box_val_y,
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(full_data, x="性別", y=box_val_y)
            st.plotly_chart(fig, use_container_width=True)

        # ログを記録
        # add_row_to_gsheet(
        #     gsheet_connector,
        #     [
        #         [
        #             datetime.datetime.now(
        #                 datetime.timezone(datetime.timedelta(hours=9))
        #             ).strftime("%Y-%m-%d %H:%M:%S"),
        #             st.session_state.username,
        #             "BoxPlot",
        #             box_val_y,
        #             "-",
        #             "-",
        #         ]
        #     ],
        # )


# ---------------- Simple Regression ----------------------------------
def lr():
    st.title("Predict with Simple Regression")
    df = load_full_data()

    st.sidebar.markdown("## Default Type1！")

    # sidebar でグラフを選択
    df_type = st.sidebar.radio("", ("Type1", "Type2", "Type3"))

    # Type1; フルデータ
    if df_type == "Type1":
        filtered_df = load_num_data()
    # Type2: 女子のみのデータ
    elif df_type == "Type2":
        filtered_df = d.load_filtered_data(df, "女子")
    # Type3: 高1女子のみのデータ
    else:
        filtered_df = d.load_filtered_data(df, "高1女子")

    # 変数を取得してから、回帰したい
    with st.form("get_lr_data"):
        y_label = st.selectbox("Prediction(TargetVariable)", X_COLS)
        x_label = st.selectbox("ExplanatoryVariable", X_COLS)

        # trainとtestをsplit
        df_train = pd.concat(
            [
                filtered_df[filtered_df.no < TEST_START_INDEX],
                filtered_df[filtered_df.no > TEST_END_INDEX],
            ]
        )
        df_test = pd.concat(
            [
                filtered_df[TEST_START_INDEX <= filtered_df.no],
                filtered_df[filtered_df.no <= TEST_END_INDEX],
            ]
        )

        y_train = df_train[[y_label]]
        y_test = df_test[[y_label]]
        X_train = df_train[[x_label]]
        X_test = df_test[[x_label]]

        submitted_lr = st.form_submit_button("run Simple Regression")

        if submitted_lr:
            # モデルの構築
            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)
            y_pred = model_lr.predict(X_test)

            # ログを記録
            # add_row_to_gsheet(
            #     gsheet_connector,
            #     [
            #         [
            #             datetime.datetime.now(
            #                 datetime.timezone(datetime.timedelta(hours=9))
            #             ).strftime("%Y-%m-%d %H:%M:%S"),
            #             st.session_state.username,
            #             "Simple Regression",
            #             y_label,
            #             x_label,
            #             "-",
            #         ]
            #     ],
            # )

            # グラフの描画
            fig = px.scatter(
                x=filtered_df[x_label].values,
                y=filtered_df[y_label].values,
                labels={"x": x_label, "y": y_label},
                trendline="ols",
                trendline_color_override="red",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------- Multiple Regression ----------------------------------
def multi_lr():
    st.title("Predict with Multiple Regression")
    df = load_full_data()

    st.sidebar.markdown("## Default Type1！")

    # sidebar でグラフを選択
    df_type = st.sidebar.radio("", ("Type1", "Type2", "Type3"))

    # Type1; フルデータ
    if df_type == "Type1":
        filtered_df = load_num_data()
    # Type2: 女子のみのデータ
    elif df_type == "Type2":
        filtered_df = d.load_filtered_data(df, "女子")
    # Type3: 高1女子のみのデータ
    else:
        filtered_df = d.load_filtered_data(df, "高1女子")

    # 変数を取得してから、回帰したい
    with st.form("get_lr_data"):
        y_label = st.selectbox("Prediction(TargetVariable)", X_COLS)
        x_labels = st.multiselect("ExplanatoryVariable", X_COLS)

        # trainとtestをsplit
        df_train = pd.concat(
            [
                filtered_df[filtered_df.no < TEST_START_INDEX],
                filtered_df[filtered_df.no > TEST_END_INDEX],
            ]
        )
        df_test = pd.concat(
            [
                filtered_df[TEST_START_INDEX <= filtered_df.no],
                filtered_df[filtered_df.no <= TEST_END_INDEX],
            ]
        )

        y_train = df_train[[y_label]]
        y_test = df_test[[y_label]]
        X_train = df_train[x_labels]
        X_test = df_test[x_labels]

        submitted_multi = st.form_submit_button("run Multiple Regression")

        if submitted_multi:
            ## エラー対応
            if len(x_labels) == 0:
                st.markdown("### 予測に使いたい変数を1つ以上選んでください！")

            else:
                # モデルの構築
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)
                y_pred = model_lr.predict(X_test)

                # ログを記録
                # add_row_to_gsheet(
                #     gsheet_connector,
                #     [
                #         [
                #             datetime.datetime.now(
                #                 datetime.timezone(datetime.timedelta(hours=9))
                #             ).strftime("%Y-%m-%d %H:%M:%S"),
                #             st.session_state.username,
                #             "Multiple Regression",
                #             y_label,
                #             "_".join(x_labels),
                #             "-",
                #         ]
                #     ],
                # )

                # 結果の表示
                coef = model_lr.coef_[0]
                intercept = model_lr.intercept_[0]

                ans = "##### " + y_label + " = "
                for c, label in zip(coef, x_labels):
                    ans += " **{:.2f}** x **{}の値** +".format(c, label)

                # 切片を追記
                if intercept > 0:
                    ans += str(round(intercept, 3))
                else:
                    ans = ans[:-1] + "- " + str(round(abs(intercept), 3))
                st.markdown(ans)

                st.markdown("花子の他の測定値は以下の通り")
                st.table(target_df)

                st.markdown("上記の式に、データを当てはめると....")

                pred_str = "##### " + y_label + " = "
                pred_ans = 0
                for c, label in zip(coef, x_labels):
                    pred_str += " **{:.2f}** x **{}** +".format(
                        c, target_df.at[0, label]
                    )
                    pred_ans += round(c, 3) * target_df.at[0, label]

                # 切片を追記
                if intercept > 0:
                    pred_str += str(round(intercept, 3))
                else:
                    pred_str = pred_str[:-1] + "- " + str(round(abs(intercept), 3))

                pred_ans += round(intercept, 3)

                st.markdown(pred_str)
                st.success("予測結果：" + str(round(pred_ans, 3)))

                st.write("※ 決定係数： " + str(round(model_lr.score(X_train, y_train), 3)))

                # グラフの描画
                # plot_y = list(map(lambda y: y[0], y_pred))
                # fig = px.scatter(
                #     x=y_test[y_label].values, y=plot_y, labels={"x": "実測値", "y": "予測値"}
                # )
                # st.plotly_chart(fig, use_container_width=True)


#### main contents
# gsheet_connector = connect_to_gsheet()
main()
