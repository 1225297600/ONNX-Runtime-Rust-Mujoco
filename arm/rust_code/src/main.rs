/*
    1.参考microduck
    2.使用ONNX
    3.使用微软ONNX Runtime库（ORT）
    4.
*/

use std::path::Path;
use std::thread;
use std::time::Duration;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use anyhow::Context;

// 参数数量
const OBS_LEN : usize = 61;


#[derive(Clone, Copy)]
enum Module {
    AlphaStand,
    AlphaWalking,
}

fn main() -> anyhow::Result<()> {
    println!("ONNX start");

    let module = Module::AlphaStand;
    let alpha_stand_model_path = Path::new("alpha_stand.onnx");
    let alpha_walking_model_path = Path::new("alpha_walking.onnx");

    // 1.打开模型文件，建立回话
    let mut session_alpha_stand = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?   // Level1 基础 / Level2 通用扩展 / Level3 所有优化全开
        .with_intra_threads(1)?                                     // 一个算子内部开几个线程
        .with_inter_threads(1)?                                     // 算子与算子之间的并行度
        .commit_from_file(alpha_stand_model_path)                  // 模型加载路径
        .with_context(|| alpha_stand_model_path.display().to_string())?;        // 错误日志，显示模型信息
    let mut session_alpha_walking = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file(alpha_walking_model_path)
        .with_context(|| alpha_walking_model_path.display().to_string())?;
    
    println!("ONNX loop");
    loop {
        // 2.准备输入
        let observation = vec![0.0f32; OBS_LEN];
        // 其他步骤....
        let input = Value::from_array(([1usize, OBS_LEN], observation))?;

        // 3.选择策略
        let outputs = match module{
            // 4.喂数据进去，拿结果出来
            Module::AlphaStand => session_alpha_stand
                                    .run(ort::inputs!["obs" => &input])?,
            Module::AlphaWalking => session_alpha_walking
                                    .run(ort::inputs!["obs" => &input])?,
        };

        // 5.应用输出
        // 其他步骤....
        let (_shape, _data) = outputs[0].try_extract_tensor::<f32>()?;

        // 6.等待下一次执行
        thread::sleep(Duration::from_millis(20));
    }

}
